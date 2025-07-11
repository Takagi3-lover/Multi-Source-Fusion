# multi_source_fusion/__init__.py

"""
基于深度学习的多源组合定位系统

该系统实现了MTINN-EKF-MM混合定位框架，结合了：
- 多任务物理信息神经网络 (MTINN)
- 扩展卡尔曼滤波 (EKF)
- 地图匹配 (Map Matching)

用于重载列车的高精度定位。
"""

__version__ = "1.0.0"
__author__ = "Multi-Source Fusion"

# 导入主要组件
from .core import config, CoordinateSystem, NavFrame, SystemState
from .modules import MTINN, MTINNLoss, MTINN_EKF, MapMatcher
from .inference import MTINNPredictor
from .training import train

__all__ = [
    'config', 'CoordinateSystem', 'NavFrame', 'SystemState',
    'MTINN', 'MTINNLoss', 'MTINN_EKF', 'MapMatcher',
    'MTINNPredictor', 'train'
]