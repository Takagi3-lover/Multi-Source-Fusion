# multi_source_fusion/modules/mtinn_model.py

import torch
import torch.nn as nn
from typing import Tuple

from core.config import config


class MTINN(nn.Module):
    """
    多任务物理信息神经网络 (Multi-Task Physics-Informed Neural Network)。
    该模型预测状态变化量，而不是绝对状态。
    输出: [姿态变化, 速度变化, 位置变化]
    """

    def __init__(self, input_size=7, hidden_size=64, output_att_size=3, output_vel_size=3, output_pos_size=3,
                 num_layers=1):
        super(MTINN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        dropout_rate = config.get('mtinn_hyperparams.dropout_rate', 0.2)

        print("MTINN模型配置:")
        print("  - 输出类型: 状态变化量 (delta)")
        print("  - 姿态变化: [delta_roll, delta_pitch, delta_yaw]")
        print("  - 速度变化: [delta_v_e, delta_v_n, delta_v_u]")
        print("  - 位置变化: [delta_lat, delta_lon, delta_h]")

        # 1. 共享LSTM层
        self.shared_lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )

        # 2. 任务特定层 - 简化架构
        self.attitude_lstm_cell = nn.LSTMCell(hidden_size + input_size, hidden_size // 2)
        self.velocity_lstm_cell = nn.LSTMCell(hidden_size + output_att_size + input_size, hidden_size // 2)
        self.position_lstm_cell = nn.LSTMCell(hidden_size + output_vel_size, hidden_size // 2)

        # 3. 输出层 - 输出变化量，使用tanh激活函数限制输出范围
        self.fc_attitude = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 4, output_att_size),
        )

        self.fc_velocity = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 4, output_vel_size),
        )

        self.fc_position = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 4, output_pos_size),
        )

        # Dropout层
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        处理一个输入序列，输出状态变化量。

        Args:
            x (torch.Tensor): 输入的传感器数据序列，形状为 (batch_size, seq_len, input_size)。

        Returns:
            torch.Tensor: 预测的状态变化量，形状为 (batch_size, seq_len, 9)。
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # 初始化隐藏状态
        shared_h, shared_c = self._init_shared_hidden(batch_size, device)
        att_h, att_c = self._init_task_hidden(batch_size, device)
        vel_h, vel_c = self._init_task_hidden(batch_size, device)
        pos_h, pos_c = self._init_task_hidden(batch_size, device)

        outputs = []

        # 1. 共享LSTM处理
        shared_out, (shared_h_final, shared_c_final) = self.shared_lstm(x, (shared_h, shared_c))
        shared_out = self.dropout(shared_out)

        # 2. 逐时间步处理
        for t in range(seq_len):
            current_shared_out = shared_out[:, t, :]
            current_input = x[:, t, :]

            # 姿态变化预测
            att_input = torch.cat([current_shared_out, current_input], dim=1)
            att_h, att_c = self.attitude_lstm_cell(att_input, (att_h, att_c))
            att_h_dropped = self.dropout(att_h)
            delta_att = self.fc_attitude(att_h_dropped)

            # 速度变化预测
            vel_input = torch.cat([current_shared_out, delta_att.detach(), current_input], dim=1)
            vel_h, vel_c = self.velocity_lstm_cell(vel_input, (vel_h, vel_c))
            vel_h_dropped = self.dropout(vel_h)
            delta_vel = self.fc_velocity(vel_h_dropped)

            # 位置变化预测
            pos_input = torch.cat([current_shared_out, delta_vel.detach()], dim=1)
            pos_h, pos_c = self.position_lstm_cell(pos_input, (pos_h, pos_c))
            pos_h_dropped = self.dropout(pos_h)
            delta_pos = self.fc_position(pos_h_dropped)

            # 合并输出 - 这些都是变化量
            combined_delta = torch.cat([delta_att, delta_vel, delta_pos], dim=1)
            outputs.append(combined_delta)

        return torch.stack(outputs, dim=1)

    def _init_shared_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """初始化共享LSTM的隐藏状态"""
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device),
                torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device))

    def _init_task_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """初始化任务特定LSTMCell的隐藏状态"""
        return (torch.zeros(batch_size, self.hidden_size // 2, device=device),
                torch.zeros(batch_size, self.hidden_size // 2, device=device))