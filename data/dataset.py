# multi_source_fusion/data/dataset.py

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import List, Optional

from . import loaders
from ..core.config import config
from ..core.types import NavFrame


def create_synchronized_dataset(
        imu_path: str,
        gnss_path: str,
        odo_path: str,
        ground_truth_path: Optional[str] = None
) -> List[NavFrame]:
    """
    加载所有传感器数据，并将其同步到IMU时间戳上。

    Args:
        imu_path (str): IMU数据文件路径。
        gnss_path (str): GNSS数据文件路径。
        odo_path (str): ODO数据文件路径。
        ground_truth_path (Optional[str]): 可选的地面真值文件路径。

    Returns:
        List[NavFrame]: 同步后的导航数据帧列表。
    """
    # 1. 加载数据
    print("正在加载数据...")
    imu_df = loaders.load_imu_data(imu_path)
    gnss_df = loaders.load_gnss_data(gnss_path)
    odo_df = loaders.load_odo_data(odo_path)

    if imu_df.empty:
        raise ValueError("IMU数据加载失败，无法继续。")

    # 将所有数据的时间戳设置为索引，便于插值
    imu_df.set_index('timestamp', inplace=True)
    if not gnss_df.empty:
        gnss_df.set_index('timestamp', inplace=True)
    if not odo_df.empty:
        odo_df.set_index('timestamp', inplace=True)

    # 2. 合并与插值
    print("正在同步数据...")

    # 创建一个基于IMU时间戳的空DataFrame
    synced_df = pd.DataFrame(index=imu_df.index)

    # 添加IMU数据
    for col in imu_df.columns:
        synced_df[col] = imu_df[col]

    # 添加GNSS数据（如果存在）
    if not gnss_df.empty:
        for col in gnss_df.columns:
            synced_df[col] = gnss_df[col]

    # 添加ODO数据（如果存在）
    if not odo_df.empty:
        for col in odo_df.columns:
            synced_df[col] = odo_df[col]

    # 线性插值填补低频传感器数据的空白
    synced_df.interpolate(method='linear', limit_direction='both', inplace=True)

    # 填充剩余的NaN值
    synced_df.fillna(method='bfill', inplace=True)
    synced_df.fillna(method='ffill', inplace=True)

    # 3. 创建NavFrame对象列表
    print("正在创建导航数据帧...")
    nav_frames: List[NavFrame] = []

    for timestamp, row in synced_df.iterrows():
        # 创建IMU数据字典
        imu_data = {
            'ax': row.get('ax', 0.0), 'ay': row.get('ay', 0.0), 'az': row.get('az', 0.0),
            'gx': row.get('gx', 0.0), 'gy': row.get('gy', 0.0), 'gz': row.get('gz', 0.0)
        }

        # 创建ODO数据字典
        odo_data = {'velocity': row.get('velocity', 0.0)}

        # 创建GNSS数据字典（如果存在）
        gnss_data = None
        if not gnss_df.empty and all(col in row.index for col in ['lat', 'lon', 'h']):
            gnss_data = {
                'lat': row.get('lat', 0.0), 'lon': row.get('lon', 0.0), 'h': row.get('h', 0.0),
                'std_lat': row.get('std_lat', 1.0), 'std_lon': row.get('std_lon', 1.0), 'std_h': row.get('std_h', 1.0)
            }

        frame = NavFrame(
            timestamp=timestamp,
            imu_data=imu_data,
            odo_data=odo_data,
            gnss_data=gnss_data
            # ground_truth 可以在这里从一个单独的文件加载和同步
        )
        nav_frames.append(frame)

    print(f"数据同步完成，共创建 {len(nav_frames)} 个导航数据帧。")
    return nav_frames


class MTINNDataset(Dataset):
    """
    为MTINN模型准备时序数据的PyTorch Dataset。
    """

    def __init__(self, nav_frames: List[NavFrame], ground_truth_frames: Optional[List[NavFrame]] = None):
        """
        Args:
            nav_frames (List[NavFrame]): 同步后的导航数据帧列表。
            ground_truth_frames (Optional[List[NavFrame]]): 可选的地面真值帧列表。
        """
        self.nav_frames = nav_frames
        self.gt_frames = ground_truth_frames
        self.seq_len = config.get('mtinn_hyperparams.sequence_length', 50)

        # 将数据转换为numpy数组以便快速访问
        self._prepare_arrays()

    def _prepare_arrays(self):
        """将NavFrame列表转换为高效的Numpy数组，以加速__getitem__。"""
        # 输入特征: ax, ay, az, gx, gy, gz, v_odo (7维)
        self.input_data = np.array([
            [
                frame.imu_data['ax'], frame.imu_data['ay'], frame.imu_data['az'],
                frame.imu_data['gx'], frame.imu_data['gy'], frame.imu_data['gz'],
                frame.odo_data['velocity']
            ] for frame in self.nav_frames
        ], dtype=np.float32)

        # 标签/真值: lat, lon, h, v_e, v_n, v_u, roll, pitch, yaw (9维)
        if self.gt_frames:
            self.target_data = np.array([
                [
                    gt.ground_truth['lat'], gt.ground_truth['lon'], gt.ground_truth['h'],
                    gt.ground_truth['v_e'], gt.ground_truth['v_n'], gt.ground_truth['v_u'],
                    gt.ground_truth['roll'], gt.ground_truth['pitch'], gt.ground_truth['yaw']
                ] for gt in self.gt_frames
            ], dtype=np.float32)
        else:
            self.target_data = None

    def __len__(self):
        # 长度是可生成的独立序列的数量
        return max(0, len(self.nav_frames) - self.seq_len + 1)

    def __getitem__(self, idx):
        """
        返回一个序列的输入数据和对应的标签（如果存在）。
        """
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")

        input_seq = self.input_data[idx: idx + self.seq_len]

        item = {'input': torch.from_numpy(input_seq)}

        if self.target_data is not None:
            target_seq = self.target_data[idx: idx + self.seq_len]
            item['target'] = torch.from_numpy(target_seq)

        return item