# multi_source_fusion/modules/mtinn_model.py

import torch
import torch.nn as nn
from typing import Tuple, Optional

from ..core.config import config

class MTINN(nn.Module):
    """
    多任务物理信息神经网络 (Multi-Task Physics-Informed Neural Network)。
    该模型采用"共享-特定任务"的LSTM架构，以联合估计姿态、速度和位置。
    它强制执行一个因果依赖链：姿态的预测被用于速度的预测，速度的预测被用于位置的预测。
    """

    def __init__(self, input_size=7, hidden_size=128, output_att_size=3, output_vel_size=3, output_pos_size=3):
        super(MTINN, self).__init__()

        self.hidden_size = hidden_size
        dropout_rate = config.get('mtinn_hyperparams.dropout_rate', 0.2)

        # 1. 共享LSTM层: 从所有传感器输入中提取共享的时序特征
        self.shared_lstm = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=dropout_rate)

        # 2. 特定任务LSTM单元 (LSTMCell)
        # 使用LSTMCell可以更明确地控制每一步的输入和隐藏状态，便于实现因果链。

        # 姿态任务层: 输入 = 共享特征 + 上一时刻姿态预测 + 原始传感器输入
        self.attitude_lstm_cell = nn.LSTMCell(hidden_size + output_att_size + input_size, hidden_size)

        # 速度任务层: 输入 = 共享特征 + 上一时刻速度预测 + 当前姿态预测 + 原始传感器输入
        self.velocity_lstm_cell = nn.LSTMCell(hidden_size + output_vel_size + output_att_size + input_size, hidden_size)

        # 位置任务层: 输入 = 共享特征 + 上一时刻位置预测 + 当前速度预测 + 原始传感器输入
        self.position_lstm_cell = nn.LSTMCell(hidden_size + output_pos_size + output_vel_size + input_size, hidden_size)

        # 3. 全连接层 (FC): 将任务特定的隐藏状态映射到最终输出
        self.fc_attitude = nn.Linear(hidden_size, output_att_size)
        self.fc_velocity = nn.Linear(hidden_size, output_vel_size)
        self.fc_position = nn.Linear(hidden_size, output_pos_size)

        # Dropout层用于MC Dropout不确定性估计
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        处理一个输入序列。

        Args:
            x (torch.Tensor): 输入的传感器数据序列，形状为 (batch_size, seq_len, input_size)。

        Returns:
            torch.Tensor: 拼接后的预测结果，形状为 (batch_size, seq_len, 9)。
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # 初始化隐藏状态和单元状态
        shared_h, shared_c = self._init_hidden(batch_size, device)
        att_h, att_c = self._init_hidden(batch_size, device)
        vel_h, vel_c = self._init_hidden(batch_size, device)
        pos_h, pos_c = self._init_hidden(batch_size, device)

        # 初始化上一时刻的预测输出为零
        prev_att = torch.zeros(batch_size, 3, device=device)
        prev_vel = torch.zeros(batch_size, 3, device=device)
        prev_pos = torch.zeros(batch_size, 3, device=device)

        outputs = []

        # 1. 首先通过共享LSTM处理整个序列
        shared_out, (shared_h_final, shared_c_final) = self.shared_lstm(x, (shared_h, shared_c))
        shared_out = self.dropout(shared_out)

        # 2. 逐个时间步处理，以强制执行因果链
        for t in range(seq_len):
            # 获取当前时间步的共享层输出和原始输入
            current_shared_out = shared_out[:, t, :]
            current_input = x[:, t, :]

            # --- 姿态任务 ---
            att_input = torch.cat([current_shared_out, prev_att, current_input], dim=1)
            att_h, att_c = self.attitude_lstm_cell(att_input, (att_h, att_c))
            att_h_dropped = self.dropout(att_h)
            p_t = self.fc_attitude(att_h_dropped)

            # --- 速度任务 ---
            vel_input = torch.cat([current_shared_out, prev_vel, p_t.detach(), current_input], dim=1)
            vel_h, vel_c = self.velocity_lstm_cell(vel_input, (vel_h, vel_c))
            vel_h_dropped = self.dropout(vel_h)
            v_t = self.fc_velocity(vel_h_dropped)

            # --- 位置任务 ---
            pos_input = torch.cat([current_shared_out, prev_pos, v_t.detach(), current_input], dim=1)
            pos_h, pos_c = self.position_lstm_cell(pos_input, (pos_h, pos_c))
            pos_h_dropped = self.dropout(pos_h)
            l_t = self.fc_position(pos_h_dropped)

            # 拼接当前时间步的输出
            combined_output = torch.cat([p_t, v_t, l_t], dim=1)
            outputs.append(combined_output)

            # 更新上一时刻的预测，用于下一个时间步
            prev_att, prev_vel, prev_pos = p_t, v_t, l_t

        # 将输出列表堆叠成一个张量
        return torch.stack(outputs, dim=1)

    def _init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """初始化LSTM的隐藏状态和单元状态为零。"""
        return (torch.zeros(batch_size, self.hidden_size, device=device),
                torch.zeros(batch_size, self.hidden_size, device=device))