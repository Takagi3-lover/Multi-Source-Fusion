# multi_source_fusion/data/loaders.py

import pandas as pd
import numpy as np
from typing import Optional
from ..core.config import config


def load_imu_data(file_path: str) -> pd.DataFrame:
    """
    从*.txt文件加载IMU数据。
    文件格式：7列，空格分隔 (GNSS周秒, 增量角x/y/z, 增量速度x/y/z)
    将增量值转换为速率值（角速度 rad/s, 加速度 m/s^2）。
    """
    imu_cols = ['timestamp', 'd_angle_x', 'd_angle_y', 'd_angle_z', 'd_vel_x', 'd_vel_y', 'd_vel_z']
    try:
        df = pd.read_csv(file_path, sep=r'\s+', header=None, names=imu_cols, comment='%')

        if df.empty:
            print(f"警告: IMU数据文件 '{file_path}' 为空")
            return pd.DataFrame()

    except FileNotFoundError:
        print(f"错误: IMU数据文件未找到于 '{file_path}'")
        return pd.DataFrame()
    except Exception as e:
        print(f"错误: 读取IMU数据文件时出错: {e}")
        return pd.DataFrame()

    # 计算时间间隔 dt，第一行使用配置文件中的频率
    dt = df['timestamp'].diff()
    # 使用配置中的IMU频率填充第一个NaN值
    dt.iloc[0] = 1.0 / config.get('sensor_params.imu.rate_hz', 100.0)

    # 将增量转换为速率
    # 陀螺仪输出：角速度 (rad/s)
    df['gx'] = df['d_angle_x'] / dt
    df['gy'] = df['d_angle_y'] / dt
    df['gz'] = df['d_angle_z'] / dt

    # 加速度计输出：加速度 (m/s^2)
    df['ax'] = df['d_vel_x'] / dt
    df['ay'] = df['d_vel_y'] / dt
    df['az'] = df['d_vel_z'] / dt

    # 返回包含时间戳和速率的DataFrame
    return df[['timestamp', 'gx', 'gy', 'gz', 'ax', 'ay', 'az']]


def load_gnss_data(file_path: str) -> pd.DataFrame:
    """
    从*.pos文件加载GNSS定位结果。
    文件格式：7列，空格分隔 (GNSS周秒, 纬度, 经度, 高程, 纬度标准差, 经度标准差, 高程标准差)
    """
    gnss_cols = ['timestamp', 'lat', 'lon', 'h', 'std_lat', 'std_lon', 'std_h']
    try:
        df = pd.read_csv(file_path, sep=r'\s+', header=None, names=gnss_cols, comment='%')

        if df.empty:
            print(f"警告: GNSS数据文件 '{file_path}' 为空")
            return pd.DataFrame()

        # 将度转换为弧度以便内部计算
        df['lat'] = np.radians(df['lat'])
        df['lon'] = np.radians(df['lon'])
        return df

    except FileNotFoundError:
        print(f"错误: GNSS数据文件未找到于 '{file_path}'")
        return pd.DataFrame()
    except Exception as e:
        print(f"错误: 读取GNSS数据文件时出错: {e}")
        return pd.DataFrame()


def load_odo_data(file_path: str) -> pd.DataFrame:
    """
    从*.txt文件加载ODO速度数据。
    文件格式：2列，空格分隔 (时间戳, 速度)
    """
    odo_cols = ['timestamp', 'velocity']
    try:
        df = pd.read_csv(file_path, sep=r'\s+', header=None, names=odo_cols, comment='%')

        if df.empty:
            print(f"警告: ODO数据文件 '{file_path}' 为空")
            return pd.DataFrame()

        return df

    except FileNotFoundError:
        print(f"错误: ODO数据文件未找到于 '{file_path}'")
        return pd.DataFrame()
    except Exception as e:
        print(f"错误: 读取ODO数据文件时出错: {e}")
        return pd.DataFrame()


def load_map_data(file_path: str) -> pd.DataFrame:
    """
    从*.nav文件加载铁路线路地图数据。
    该函数会尝试使用不同的分隔符（逗号、空格）来解析文件。
    """
    try:
        # 尝试以标准逗号分隔读取
        df = pd.read_csv(file_path, comment='%')

        if df.empty:
            print(f"警告: 地图数据文件 '{file_path}' 为空")
            return pd.DataFrame()

    except Exception:
        try:
            # 如果失败，尝试以空格分隔读取
            df = pd.read_csv(file_path, sep=r'\s+', comment='%', header=0)

            if df.empty:
                print(f"警告: 地图数据文件 '{file_path}' 为空")
                return pd.DataFrame()

        except FileNotFoundError:
            print(f"错误: 地图数据文件未找到于 '{file_path}'")
            return pd.DataFrame()
        except Exception as e:
            print(f"错误: 无法解析地图文件 '{file_path}': {e}")
            return pd.DataFrame()

    # 根据需求文档，重命名一些关键列以保持一致性
    # 这里假设了原始文件中的列名，可能需要根据实际文件进行调整
    rename_map = {
        '纬度': 'lat', '经度': 'lon', '大地高度': 'h', '方位角': 'azimuth'
    }
    df.rename(columns=lambda c: rename_map.get(c.strip(), c.strip().lower()), inplace=True)

    # 将度转换为弧度
    if 'lat' in df.columns:
        df['lat'] = np.radians(df['lat'])
    if 'lon' in df.columns:
        df['lon'] = np.radians(df['lon'])

    return df