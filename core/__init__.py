# multi_source_fusion/core/__init__.py

"""
核心模块，包含配置管理、坐标系变换和数据类型定义。
"""

from .config import config
from .coordinates import CoordinateSystem
from .types import NavFrame, SystemState

__all__ = ['config', 'CoordinateSystem', 'NavFrame', 'SystemState']