# multi_source_fusion/data/loaders.py

import numpy as np
import pandas as pd

from core.config import config

def load_imu_data(file_path: str) -> pd.DataFrame:
    """
    修改后: 直接加载IMU增量数据，不转换为速率。
    文件格式：7列，(GNSS周秒, 增量角x/y/z, 增量速度x/y/z)
    """
    imu_cols = ['timestamp', 'd_angle_x', 'd_angle_y', 'd_angle_z', 'd_vel_x', 'd_vel_y', 'd_vel_z']
    try:
        df = pd.read_csv(file_path, sep=r'\s+', header=None, names=imu_cols, comment='%')

        if df.empty:
            print(f"警告: IMU数据文件 '{file_path}' 为空")
            return pd.DataFrame()
        df.dropna(inplace=True)
        print(f"IMU数据加载完成，有效数据点: {len(df)}")
        return df

    except FileNotFoundError:
        print(f"错误: IMU数据文件未找到于 '{file_path}'")
        return pd.DataFrame()
    except Exception as e:
        print(f"错误: 读取IMU数据文件时出错: {e}")
        return pd.DataFrame()

def load_gnss_data(file_path: str) -> pd.DataFrame:
    """
    从*.pos文件加载GNSS定位结果。
    文件格式：7列，(GNSS周秒, 纬度, 经度, 高程, 纬度标准差, 经度标准差, 高程标准差)
    """
    gnss_cols = ['timestamp', 'lat', 'lon', 'h', 'std_lat', 'std_lon', 'std_h']
    try:
        df = pd.read_csv(file_path, sep=r'\s+', header=None, names=gnss_cols, comment='%')

        if df.empty:
            print(f"警告: GNSS数据文件 '{file_path}' 为空")
            return pd.DataFrame()
        df.dropna(inplace=True)

        # 将度转换为弧度
        df['lat'] = np.radians(df['lat'])
        df['lon'] = np.radians(df['lon'])

        print(f"GNSS数据加载完成，有效数据点: {len(df)}")
        return df

    except FileNotFoundError:
        print(f"错误: GNSS数据文件未找到于 '{file_path}'")
        return pd.DataFrame()
    except Exception as e:
        print(f"错误: 读取GNSS数据文件时出错: {e}")
        return pd.DataFrame()