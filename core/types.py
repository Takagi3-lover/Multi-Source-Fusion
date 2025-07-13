# multi_source_fusion/core/types.py

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

@dataclass
class NavFrame:
    """
    统一的导航数据帧结构。
    用于在数据加载和同步后，封装单个时间戳的所有传感器信息。
    """
    timestamp: float  # GNSS周秒
    imu_data: Optional[Dict[str, float]] = None  # {'ax', 'ay', 'az', 'gx', 'gy', 'gz'}
    odo_data: Optional[Dict[str, float]] = None  # {'velocity'} 或 None 表示无里程计数据
    gnss_data: Optional[Dict[str, float]] = None  # {'lat', 'lon', 'h', 'std_lat', 'std_lon', 'std_h'}
    ground_truth: Optional[Dict[str, float]] = None  # {'lat', 'lon', 'h', 'v_e', 'v_n', 'v_u', 'roll', 'pitch', 'yaw'}

@dataclass
class SystemState:
    """
    系统状态向量及其协方差的数据结构。
    用于在EKF模块中表示和传递系统的完整状态。
    """
    timestamp: float

    # 状态向量 x
    attitude: np.ndarray  # [roll, pitch, yaw] in radians
    velocity: np.ndarray  # [Ve, Vn, Vu] in m/s (ENU frame)
    position: np.ndarray  # [lat, lon, h] in radians, radians, meters

    # 协方差矩阵 P (支持不同尺寸)
    covariance: np.ndarray

    def __post_init__(self):
        """确保所有numpy数组具有正确的形状。"""
        self.attitude = np.asarray(self.attitude).reshape(3)
        self.velocity = np.asarray(self.velocity).reshape(3)
        self.position = np.asarray(self.position).reshape(3)

        # 协方差矩阵自适应尺寸
        self.covariance = np.asarray(self.covariance)
        if self.covariance.ndim == 1:
            # 如果是一维数组，假设是对角元素
            size = int(np.sqrt(len(self.covariance)))
            if size * size == len(self.covariance):
                self.covariance = np.diag(self.covariance)
            else:
                # 如果不是完全平方数，可能是展平的矩阵
                size = int(np.sqrt(len(self.covariance)))
                self.covariance = self.covariance.reshape(size, size)
        elif self.covariance.ndim == 2:
            # 已经是二维矩阵，保持原样
            pass
        else:
            raise ValueError(f"协方差矩阵维度错误: {self.covariance.shape}")

    def get_main_state_covariance(self) -> np.ndarray:
        """
        获取主要状态（姿态、速度、位置）的9x9协方差矩阵。
        如果原协方差矩阵更大，则提取相关子矩阵。
        """
        if self.covariance.shape == (9, 9):
            return self.covariance
        elif self.covariance.shape == (21, 21):
            # 从21x21矩阵中提取位置、速度、姿态的协方差 (前9个状态)
            indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # pos(3) + vel(3) + att(3)
            return self.covariance[np.ix_(indices, indices)]
        else:
            # 对于其他尺寸，返回前9x9子矩阵
            size = min(9, self.covariance.shape[0])
            return self.covariance[:size, :size]