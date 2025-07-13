# multi_source_fusion/core/types.py

from dataclasses import dataclass, field
import numpy as np

# 移植自 common/types.h
@dataclass
class IMU:
    time: float = 0.0
    dt: float = 0.0
    dtheta: np.ndarray = field(default_factory=lambda: np.zeros(3)) # 角度增量
    dvel: np.ndarray = field(default_factory=lambda: np.zeros(3))   # 速度增量

@dataclass
class GNSS:
    time: float = 0.0
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))    # lat, lon, h (rad, rad, m)
    std: np.ndarray = field(default_factory=lambda: np.zeros(3))    # std for pos (m, m, m)

# 移植自 kf_gins_types.h
@dataclass
class Attitude:
    qbn: np.ndarray = field(default_factory=lambda: np.array([1.0, 0, 0, 0])) # w,x,y,z
    cbn: np.ndarray = field(default_factory=lambda: np.eye(3))               # b-frame to n-frame
    euler: np.ndarray = field(default_factory=lambda: np.zeros(3))           # roll, pitch, yaw (rad)

@dataclass
class PVA:
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))    # lat, lon, h (rad, rad, m)
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))    # v_n, v_e, v_d (m/s) -> 注意：KF-GINS内部使用NED坐标系
    att: Attitude = field(default_factory=Attitude)

@dataclass
class ImuError:
    gyrbias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    accbias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyrscale: np.ndarray = field(default_factory=lambda: np.zeros(3))
    accscale: np.ndarray = field(default_factory=lambda: np.zeros(3))

@dataclass
class NavState:
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    euler: np.ndarray = field(default_factory=lambda: np.zeros(3))
    imuerror: ImuError = field(default_factory=ImuError)