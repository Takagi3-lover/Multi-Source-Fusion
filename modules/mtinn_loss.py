# multi_source_fusion/modules/mtinn_loss.py

import torch
import torch.nn as nn
import numpy as np

from ..core.config import config
from ..core.coordinates import CoordinateSystem


class MTINNLoss(nn.Module):
    """
    MTINN的混合损失函数。
    结合了数据损失 (L_data) 和物理损失 (L_physics)。
    """

    def __init__(self, adaptive_weights=True):
        super(MTINNLoss, self).__init__()
        self.data_loss_fn = nn.MSELoss()
        self.adaptive_weights = adaptive_weights

        # 物理损失项共9个: 3(att) + 3(vel_imu) + 3(pos)
        # ODO约束直接施加在速度上，不单独作为损失项
        num_physics_losses = 9

        if self.adaptive_weights:
            # 可学习的参数，用于自动平衡各个损失项的权重
            self.log_vars = nn.Parameter(torch.zeros(num_physics_losses + 1))  # +1 for data_loss

    def forward(self, predictions, inputs, dt, targets=None):
        """
        计算总损失。

        Args:
            predictions (torch.Tensor): 模型输出 (batch, seq_len, 9)
            inputs (torch.Tensor): 模型输入 (batch, seq_len, 7)
            dt (float): 时间间隔
            targets (torch.Tensor, optional): 地面真值 (batch, seq_len, 9)

        Returns:
            torch.Tensor: 加权后的总损失。
        """
        device = predictions.device

        # 1. 数据损失 (L_data)
        loss_data = torch.tensor(0.0, device=device)
        if targets is not None:
            loss_data = self.data_loss_fn(predictions, targets)

        # 2. 物理损失 (L_physics)
        # 使用有限差分近似导数: dy/dt ≈ (y_t - y_{t-1}) / dt
        # 我们需要 t 和 t-1 时刻的预测值
        if predictions.size(1) < 2:
            # 如果序列长度小于2，无法计算导数，只返回数据损失
            return loss_data

        pred_t = predictions[:, 1:, :]
        pred_t_minus_1 = predictions[:, :-1, :]

        # 同样需要 t 和 t-1 时刻的输入值
        inputs_t = inputs[:, 1:, :]

        # 计算导数
        derivatives = (pred_t - pred_t_minus_1) / dt

        # --- 姿态物理损失 ---
        att_pred = pred_t[:, :, :3]  # roll, pitch, yaw
        att_dot_pred = derivatives[:, :, :3]

        # 从输入中获取角速度
        omega = inputs_t[:, :, 3:6]  # gx, gy, gz

        loss_phy_att = self._compute_attitude_loss(att_dot_pred, att_pred, omega)

        # --- 速度物理损失 ---
        vel_pred = pred_t[:, :, 3:6]  # Ve, Vn, Vu
        vel_dot_pred = derivatives[:, :, 3:6]

        # 从输入中获取加速度和ODO速度
        accel = inputs_t[:, :, :3]  # ax, ay, az
        v_odo = inputs_t[:, :, 6].unsqueeze(-1)  # v_odo

        loss_phy_vel_imu, loss_phy_vel_odo = self._compute_velocity_loss(vel_dot_pred, vel_pred, att_pred, accel, v_odo)

        # --- 位置物理损失 ---
        pos_pred = pred_t[:, :, 6:9]  # lat, lon, h
        pos_dot_pred = derivatives[:, :, 6:9]

        loss_phy_pos = self._compute_position_loss(pos_dot_pred, pos_pred, vel_pred)

        # 组合所有物理损失
        all_losses = torch.cat([
            loss_data.unsqueeze(0),
            loss_phy_att.flatten(),
            loss_phy_vel_imu.flatten(),
            loss_phy_pos.flatten()
        ])

        # 3. 加权总损失
        if self.adaptive_weights:
            # L_total = sum(exp(-s_i) * L_i + s_i)
            total_loss = torch.sum(torch.exp(-self.log_vars) * all_losses + self.log_vars)
        else:
            total_loss = torch.sum(all_losses)

        return total_loss

    def _compute_attitude_loss(self, att_dot, att, omega):
        """计算姿态物理损失"""
        phi, theta, psi = att[:, :, 0], att[:, :, 1], att[:, :, 2]
        d_phi, d_theta, d_psi = att_dot[:, :, 0], att_dot[:, :, 1], att_dot[:, :, 2]
        wx, wy, wz = omega[:, :, 0], omega[:, :, 1], omega[:, :, 2]

        c_phi, s_phi = torch.cos(phi), torch.sin(phi)
        c_theta, s_theta = torch.cos(theta), torch.sin(theta)
        t_theta = torch.tan(theta)

        # 避免 cos(theta) 接近0时 tan(theta) 爆炸
        epsilon = 1e-8
        c_theta_safe = torch.clamp(c_theta, min=epsilon)

        # 姿态微分方程的残差
        res_phi = d_phi - wx - (s_phi / c_theta_safe * wy + c_phi / c_theta_safe * wz)
        res_theta = d_theta - (c_phi * wy - s_phi * wz)
        res_psi = d_psi - (s_phi * t_theta * wy + c_phi * t_theta * wz)

        return torch.stack([
            torch.mean(res_phi ** 2),
            torch.mean(res_theta ** 2),
            torch.mean(res_psi ** 2)
        ])

    def _compute_velocity_loss(self, vel_dot, vel, att, accel, v_odo):
        """计算速度物理损失"""
        # 1. IMU速度物理损失
        # 这是一个简化的版本，忽略了科里奥利力和向心力，因为它们通常比传感器噪声小
        batch_size, seq_len, _ = vel.shape
        device = vel.device

        g_n = torch.tensor([0, 0, -9.80665], device=device).expand(batch_size, seq_len, 3)

        # 计算旋转矩阵 C_bn
        roll, pitch, yaw = att[:, :, 0], att[:, :, 1], att[:, :, 2]

        # 批量计算旋转矩阵
        C_bn = self._batch_rotation_matrix(roll, pitch, yaw)

        # 将b系下的加速度旋转到n系
        f_n = torch.einsum('bsij,bsj->bsi', C_bn, accel)

        vel_dot_imu_eq = f_n + g_n
        res_vel_imu = vel_dot - vel_dot_imu_eq

        loss_phy_vel_imu = torch.mean(res_vel_imu ** 2, dim=[0, 1])

        # 2. ODO速度约束损失
        # 约束预测速度在载体坐标系下的前向分量应等于ODO速度
        # V_b = C_nb * V_n
        C_nb = C_bn.transpose(-2, -1)
        V_n = vel
        V_b = torch.einsum('bsij,bsj->bsi', C_nb, V_n)

        # 载体坐标系是"右-前-上"，所以前向速度是y轴分量
        v_forward_pred = V_b[:, :, 1].unsqueeze(-1)

        res_vel_odo = v_forward_pred - v_odo
        loss_phy_vel_odo = torch.mean(res_vel_odo ** 2)

        return loss_phy_vel_imu, loss_phy_vel_odo

    def _compute_position_loss(self, pos_dot, pos, vel):
        """计算位置物理损失"""
        lat, lon, h = pos[:, :, 0], pos[:, :, 1], pos[:, :, 2]
        d_lat, d_lon, d_h = pos_dot[:, :, 0], pos_dot[:, :, 1], pos_dot[:, :, 2]
        v_e, v_n, v_u = vel[:, :, 0], vel[:, :, 1], vel[:, :, 2]

        # 获取地球曲率半径
        # 注意：这在PyTorch中直接计算会很慢，理想情况下应预计算或使用查找表
        # 这里为了简单起见，我们假设一个常数值
        R_M = 6367000.0
        R_N = 6389000.0

        # 位置微分方程的残差
        res_lat = d_lat - (v_n / (R_M + h))
        res_lon = d_lon - (v_e / ((R_N + h) * torch.cos(lat)))
        res_h = d_h - v_u

        return torch.stack([
            torch.mean(res_lat ** 2),
            torch.mean(res_lon ** 2),
            torch.mean(res_h ** 2)
        ])

    def _batch_rotation_matrix(self, roll, pitch, yaw):
        """批量计算旋转矩阵"""
        batch_size, seq_len = roll.shape
        device = roll.device

        c_phi, s_phi = torch.cos(roll), torch.sin(roll)
        c_theta, s_theta = torch.cos(pitch), torch.sin(pitch)
        c_psi, s_psi = torch.cos(yaw), torch.sin(yaw)

        # 构建旋转矩阵 C_bn
        C_bn = torch.zeros(batch_size, seq_len, 3, 3, device=device)

        # 第一行
        C_bn[:, :, 0, 0] = c_theta * c_psi
        C_bn[:, :, 0, 1] = c_theta * s_psi
        C_bn[:, :, 0, 2] = -s_theta

        # 第二行
        C_bn[:, :, 1, 0] = s_phi * s_theta * c_psi - c_phi * s_psi
        C_bn[:, :, 1, 1] = s_phi * s_theta * s_psi + c_phi * c_psi
        C_bn[:, :, 1, 2] = s_phi * c_theta

        # 第三行
        C_bn[:, :, 2, 0] = c_phi * s_theta * c_psi + s_phi * s_psi
        C_bn[:, :, 2, 1] = c_phi * s_theta * s_psi - s_phi * c_psi
        C_bn[:, :, 2, 2] = c_phi * c_theta

        return C_bn