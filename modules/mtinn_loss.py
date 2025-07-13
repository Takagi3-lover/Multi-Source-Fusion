# multi_source_fusion/modules/mtinn_loss.py

import torch
import torch.nn as nn
from core.coordinates import CoordinateSystem


class MTINNLoss(nn.Module):
    """
    MTINN的混合损失函数 - 专为预测“状态变化量”而设计。
    """

    def __init__(self, adaptive_weights=True):
        super(MTINNLoss, self).__init__()
        self.data_loss_fn = nn.MSELoss()
        self.adaptive_weights = adaptive_weights
        num_physics_losses = 9  # 3 att, 3 vel, 3 pos
        if self.adaptive_weights:
            self.log_vars = nn.Parameter(torch.zeros(num_physics_losses + 1))

    def forward(self, pred_deltas, inputs, targets, dt):
        """
        计算总损失。
        Args:
            pred_deltas (torch.Tensor): 模型预测的状态变化量 (batch, seq_len, 9)。
            inputs (torch.Tensor): 原始传感器输入 (batch, seq_len, 7)。
            targets (torch.Tensor): 真实的（来自真值的）状态变化量 (batch, seq_len, 9)。
            dt (float): 时间间隔。
        """
        # 1. 数据损失 (L_data)
        #    直接比较预测的变化量和真实的变化量。
        loss_data = self.data_loss_fn(pred_deltas, targets)

        # 2. 物理损失 (L_physics)
        #    要计算物理损失，我们需要绝对状态。
        #    我们将通过累积真实的变化量来重建绝对状态。

        # 从真实变化量重建绝对状态序列
        # 注意：这里使用的是targets，因为物理损失应该基于真实的轨迹来约束
        absolute_states = torch.cumsum(targets, dim=1)

        # 我们需要 t 和 t-1 时刻的绝对状态
        state_t = absolute_states[:, 1:, :]

        # 预测的变化率 vs. 从绝对状态计算的物理变化率
        # 预测的变化率就是模型输出 / dt
        pred_rates = pred_deltas[:, 1:, :] / dt

        # --- 物理损失计算 ---
        att_state_t = state_t[:, :, :3]
        vel_state_t = state_t[:, :, 3:6]
        pos_state_t = state_t[:, :, 6:9]

        att_rate_pred = pred_rates[:, :, :3]
        vel_rate_pred = pred_rates[:, :, 3:6]
        pos_rate_pred = pred_rates[:, :, 6:9]

        omega = inputs[:, 1:, 3:6]
        loss_phy_att = self._compute_attitude_loss(att_rate_pred, att_state_t, omega)

        # 速度和位置损失的计算与之前类似，但输入是重建后的绝对状态
        loss_phy_pos = self._compute_position_loss(pos_rate_pred, pos_state_t, vel_state_t)

        # 速度损失暂时简化（因为其依赖项复杂）
        loss_phy_vel = torch.zeros_like(loss_phy_att)

        all_losses = torch.cat([loss_data.unsqueeze(0), loss_phy_att, loss_phy_vel, loss_phy_pos])

        if self.adaptive_weights:
            total_loss = torch.sum(torch.exp(-self.log_vars) * all_losses + self.log_vars)
        else:
            total_loss = torch.sum(all_losses)

        return total_loss

    def _compute_attitude_loss(self, att_rate_pred, att_state, omega):
        phi, theta, _ = att_state.split(1, dim=-1)
        wx, wy, wz = omega.split(1, dim=-1)

        d_phi_phy = wx + torch.tan(theta).clamp(min=-1e4, max=1e4) * (torch.sin(phi) * wy + torch.cos(phi) * wz)
        d_theta_phy = torch.cos(phi) * wy - torch.sin(phi) * wz
        d_psi_phy = (torch.sin(phi) * wy + torch.cos(phi) * wz) / torch.cos(theta).clamp(min=1e-8)

        phy_rates = torch.cat([d_phi_phy, d_theta_phy, d_psi_phy], dim=-1)
        res = att_rate_pred - phy_rates
        return torch.mean(res ** 2, dim=[0, 1])

    def _compute_position_loss(self, pos_rate_pred, pos_state, vel_state):
        lat, _, h = pos_state.split(1, dim=-1)
        v_e, v_n, v_u = vel_state.split(1, dim=-1)

        sin_lat_sq = torch.sin(lat) ** 2
        den = 1 - CoordinateSystem.E2 * sin_lat_sq

        R_M = CoordinateSystem.A * (1 - CoordinateSystem.E2) / (den.sqrt() ** 3)
        R_N = CoordinateSystem.A / den.sqrt()

        d_lat_phy = v_n / (R_M + h)
        d_lon_phy = v_e / ((R_N + h) * torch.cos(lat).clamp(min=1e-8))
        d_h_phy = v_u

        phy_rates = torch.cat([d_lat_phy, d_lon_phy, d_h_phy], dim=-1)
        res = pos_rate_pred - phy_rates
        return torch.mean(res ** 2, dim=[0, 1])