# multi_source_fusion/modules/ekf.py

import numpy as np
from typing import Dict, Optional

from ..core.types import SystemState
from ..core.config import config


class MTINN_EKF:
    """
    基于MTINN的扩展卡尔曼滤波器。
    - 预测步骤由MTINN模型驱动。
    - 更新步骤融合GNSS测量值。
    """

    def __init__(self, mtinn_predictor, initial_state: SystemState):
        """
        初始化EKF滤波器。

        Args:
            mtinn_predictor: MTINN预测器实例。
            initial_state (SystemState): 初始系统状态。
        """
        self.predictor = mtinn_predictor
        self.state = initial_state

        # 从配置加载初始噪声协方差
        q_att = config.get('ekf_params.q_att', [1.0e-6, 1.0e-6, 1.0e-5])
        q_vel = config.get('ekf_params.q_vel', [1.0e-4, 1.0e-4, 1.0e-4])
        q_pos = config.get('ekf_params.q_pos', [1.0e-3, 1.0e-3, 1.0e-3])
        self.Q = np.diag(q_att + q_vel + q_pos)

        r_pos = config.get('ekf_params.r_pos', [0.25, 0.25, 1.0])
        self.R = np.diag(r_pos)

        # 测量矩阵H，用于从9维状态中选择3维位置
        self.H = np.zeros((3, 9))
        self.H[0, 6] = 1  # lat
        self.H[1, 7] = 1  # lon
        self.H[2, 8] = 1  # h

    def predict(self, imu_data: Dict[str, float], odo_data: Dict[str, float], dt: float,
                feedback_error: Optional[np.ndarray] = None):
        """
        EKF的预测步骤。

        Args:
            imu_data (Dict[str, float]): 当前IMU数据。
            odo_data (Dict[str, float]): 当前ODO数据。
            dt (float): 时间间隔。
            feedback_error (Optional[np.ndarray]): 来自地图匹配的反馈误差。
        """
        try:
            # 1. 使用MTINN预测器获取先验状态估计和过程噪声
            predicted_state_dict, Q_k = self.predictor.predict_step_with_uncertainty(
                imu_data, odo_data, self.state, feedback_error
            )

            # 2. 更新状态预测
            self.state.timestamp += dt
            self.state.attitude = predicted_state_dict['attitude']
            self.state.velocity = predicted_state_dict['velocity']
            self.state.position = predicted_state_dict['position']

            # 3. 传播协方差
            # F_k 假设为单位矩阵，因为非线性转移由MTINN处理
            # P_k|k-1 = P_{k-1} + Q_k
            # 确保Q_k的形状正确
            if Q_k.shape != (9, 9):
                print(f"警告: Q_k形状不正确 {Q_k.shape}, 使用默认Q矩阵")
                Q_k = self.Q

            self.state.covariance = self.state.covariance + Q_k

            # 确保协方差矩阵的正定性
            self.state.covariance = self._ensure_positive_definite(self.state.covariance)

        except Exception as e:
            print(f"EKF预测步骤出错: {e}")
            # 使用默认的状态传播
            self.state.timestamp += dt
            self.state.covariance = self.state.covariance + self.Q

    def update(self, gnss_data: Dict[str, float]):
        """
        EKF的更新步骤。

        Args:
            gnss_data (Dict[str, float]): GNSS测量数据，包含lat, lon, h及其标准差。
        """
        try:
            # 1. 构建测量向量 z_k 和测量噪声协方差 R_k
            z_k = np.array([gnss_data['lat'], gnss_data['lon'], gnss_data['h']])

            # 使用GNSS提供的标准差构建R_k
            R_k = np.diag([
                gnss_data.get('std_lat', 1.0) ** 2,
                gnss_data.get('std_lon', 1.0) ** 2,
                gnss_data.get('std_h', 2.0) ** 2
            ])

            # 2. 计算卡尔曼增益 K_k
            P_k_minus = self.state.covariance
            S_k = self.H @ P_k_minus @ self.H.T + R_k

            # 确保S_k可逆
            try:
                S_k_inv = np.linalg.inv(S_k)
            except np.linalg.LinAlgError:
                print("警告: S_k矩阵奇异，使用伪逆")
                S_k_inv = np.linalg.pinv(S_k)

            K_k = P_k_minus @ self.H.T @ S_k_inv

            # 3. 更新状态估计
            # 残差 y_k = z_k - H * x_k|k-1
            x_k_minus_pos = self.state.position
            y_k = z_k - x_k_minus_pos

            # 角度环绕处理 (经度)
            if y_k[1] > np.pi:
                y_k[1] -= 2 * np.pi
            if y_k[1] < -np.pi:
                y_k[1] += 2 * np.pi

            # 状态校正
            correction = K_k @ y_k

            self.state.attitude += correction[:3]
            self.state.velocity += correction[3:6]
            self.state.position += correction[6:9]

            # 姿态角度环绕处理
            self.state.attitude = self._wrap_angles(self.state.attitude)

            # 4. 更新协方差
            I = np.eye(9)
            self.state.covariance = (I - K_k @ self.H) @ P_k_minus

            # 确保协方差矩阵的正定性
            self.state.covariance = self._ensure_positive_definite(self.state.covariance)

        except Exception as e:
            print(f"EKF更新步骤出错: {e}")
            # 如果更新失败，保持当前状态不变

    def get_state(self) -> SystemState:
        """获取当前状态的副本"""
        return SystemState(
            timestamp=self.state.timestamp,
            attitude=self.state.attitude.copy(),
            velocity=self.state.velocity.copy(),
            position=self.state.position.copy(),
            covariance=self.state.covariance.copy()
        )

    def _ensure_positive_definite(self, matrix: np.ndarray, min_eigenvalue: float = 1e-12) -> np.ndarray:
        """确保矩阵是正定的"""
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
            eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
            return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        except:
            # 如果特征值分解失败，返回对角矩阵
            return np.diag(np.diag(matrix)) + min_eigenvalue * np.eye(matrix.shape[0])

    def _wrap_angles(self, angles: np.ndarray) -> np.ndarray:
        """将角度限制在[-π, π]范围内"""
        wrapped = angles.copy()
        wrapped = np.mod(wrapped + np.pi, 2 * np.pi) - np.pi
        return wrapped