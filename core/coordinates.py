# multi_source_fusion/core/coordinates.py

import numpy as np
from core.config import config

class CoordinateSystem:
    """
    提供所有坐标系变换的静态方法。
    内部导航坐标系标准化为 ENU (东-北-天)。
    载体坐标系为 RFU (右-前-上)。
    """

    # 从配置文件加载物理常数
    A = config.get('physical_params.earth_a', 6378137.0)
    E2 = config.get('physical_params.earth_e2', 0.00669438)
    EARTH_RATE = config.get('physical_params.earth_rate', 7.292115e-5)

    @staticmethod
    def get_radii(lat_rad: float) -> tuple:
        """
        根据纬度计算地球的子午圈和卯酉圈曲率半径 (RM, RN)。

        Args:
            lat_rad (float): 纬度 (单位：弧度)。

        Returns:
            tuple: (RM, RN) in meters.
        """
        sin_lat_sq = np.sin(lat_rad) ** 2
        den = 1 - CoordinateSystem.E2 * sin_lat_sq

        rm = CoordinateSystem.A * (1 - CoordinateSystem.E2) / (den ** 1.5)
        rn = CoordinateSystem.A / np.sqrt(den)

        return rm, rn

    @staticmethod
    def wgs84_to_ecef(lat_rad: float, lon_rad: float, h: float) -> np.ndarray:
        """
        将WGS-84大地坐标 (纬度, 经度, 高程) 转换为地心地固 (ECEF) 坐标。

        Args:
            lat_rad (float): 纬度 (弧度)。
            lon_rad (float): 经度 (弧度)。
            h (float): 高程 (米)。

        Returns:
            np.ndarray: ECEF坐标 [X, Y, Z] (米)。
        """
        _, rn = CoordinateSystem.get_radii(lat_rad)
        cos_lat = np.cos(lat_rad)
        sin_lat = np.sin(lat_rad)
        cos_lon = np.cos(lon_rad)
        sin_lon = np.sin(lon_rad)

        x = (rn + h) * cos_lat * cos_lon
        y = (rn + h) * cos_lat * sin_lon
        z = (rn * (1 - CoordinateSystem.E2) + h) * sin_lat

        return np.array([x, y, z])

    @staticmethod
    def ecef_to_enu(ecef_pos: np.ndarray, ref_lat_rad: float, ref_lon_rad: float, ref_h: float) -> np.ndarray:
        """
        将ECEF坐标转换为相对于参考点的局部ENU (东-北-天) 坐标。

        Args:
            ecef_pos (np.ndarray): 目标点的ECEF坐标 [X, Y, Z]。
            ref_lat_rad (float): 参考点的纬度 (弧度)。
            ref_lon_rad (float): 参考点的经度 (弧度)。
            ref_h (float): 参考点的高程 (米)。

        Returns:
            np.ndarray: 局部ENU坐标 [E, N, U] (米)。
        """
        ref_ecef = CoordinateSystem.wgs84_to_ecef(ref_lat_rad, ref_lon_rad, ref_h)
        delta_ecef = ecef_pos - ref_ecef

        cos_lat = np.cos(ref_lat_rad)
        sin_lat = np.sin(ref_lat_rad)
        cos_lon = np.cos(ref_lon_rad)
        sin_lon = np.sin(ref_lon_rad)

        # 旋转矩阵从ECEF到ENU
        rot_matrix = np.array([
            [-sin_lon, cos_lon, 0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]
        ])

        enu_pos = rot_matrix @ delta_ecef
        return enu_pos

    @staticmethod
    def get_cbn(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """
        根据欧拉角计算从载体坐标系(b)到导航坐标系(n)的旋转矩阵 C_bn。
        该实现基于需求文档中的 C_nb 矩阵，并进行转置。
        欧拉角顺序: Z(yaw)-X(pitch)-Y(roll)

        Args:
            roll (float): 横滚角 (phi) in radians.
            pitch (float): 俯仰角 (theta) in radians.
            yaw (float): 偏航角 (psi) in radians.

        Returns:
            np.ndarray: 3x3 旋转矩阵 C_bn。
        """
        c_phi, s_phi = np.cos(roll), np.sin(roll)
        c_theta, s_theta = np.cos(pitch), np.sin(pitch)
        c_psi, s_psi = np.cos(yaw), np.sin(yaw)

        # 根据需求文档中定义的 C_nb (n-frame to b-frame)
        c_nb = np.array([
            [c_phi * c_psi + s_phi * s_theta * s_psi, -c_phi * s_psi + s_phi * s_theta * c_psi, -s_phi * c_theta],
            [s_psi * c_theta, c_psi * c_theta, s_theta],
            [s_phi * c_psi - c_phi * s_theta * s_psi, -s_phi * s_psi - c_phi * s_theta * c_psi, c_phi * c_theta]
        ])

        # C_bn 是 C_nb 的转置
        return c_nb.T

    @staticmethod
    def get_cnb(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """
        根据欧拉角计算从导航坐标系(n)到载体坐标系(b)的旋转矩阵 C_nb。

        Args:
            roll (float): 横滚角 (phi) in radians.
            pitch (float): 俯仰角 (theta) in radians.
            yaw (float): 偏航角 (psi) in radians.

        Returns:
            np.ndarray: 3x3 旋转矩阵 C_nb。
        """
        c_bn = CoordinateSystem.get_cbn(roll, pitch, yaw)
        return c_bn.T