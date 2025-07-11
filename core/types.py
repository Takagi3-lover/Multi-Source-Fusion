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
    odo_data: Optional[Dict[str, float]] = None  # {'velocity'}
    gnss_data: Optional[Dict[str, float]] = None  # {'lat', 'lon', 'h', 'std_lat', 'std_lon', 'std_h'}
    ground_truth: Optional[Dict[str, float]] = None  # {'lat', 'lon', 'h', 'v_e', 'v_n', 'v_u', 'roll', 'pitch', 'yaw'}

@dataclass
class SystemState:
    """
    系统状态向量及其协方差的数据结构。
    用于在EKF模块中表示和传递系统的完整状态。
    """
    timestamp: float

    # 状态向量 x (9x1)
    attitude: np.ndarray  # [roll, pitch, yaw] in radians
    velocity: np.ndarray  # [Ve, Vn, Vu] in m/s (ENU frame)
    position: np.ndarray  # [lat, lon, h] in radians, radians, meters

    # 协方差矩阵 P (9x9)
    covariance: np.ndarray

    def __post_init__(self):
        """确保所有numpy数组具有正确的形状。"""
        self.attitude = np.asarray(self.attitude).reshape(3)
        self.velocity = np.asarray(self.velocity).reshape(3)
        self.position = np.asarray(self.position).reshape(3)
        self.covariance = np.asarray(self.covariance).reshape(9, 9)