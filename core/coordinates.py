# multi_source_fusion/core/coordinates.py

import numpy as np


class CoordinateSystem:
    """
    移植自 earth.h 和 rotation.h.
    注意: KF-GINS 使用的导航坐标系是 NED (北-东-下), 姿态欧拉角顺序是 ZYX (yaw, pitch, roll).
    """
    WGS84_WIE = 7.2921151467E-5
    WGS84_RA = 6378137.0
    WGS84_E1 = 0.0066943799901413156

    @staticmethod
    def get_radii_and_gravity(pos: np.ndarray) -> tuple:
        lat, h = pos[0], pos[2]
        sin_lat = np.sin(lat)
        sin_lat_sq = sin_lat * sin_lat

        den_sqrt = np.sqrt(1 - CoordinateSystem.WGS84_E1 * sin_lat_sq)
        rn = CoordinateSystem.WGS84_RA / den_sqrt
        rm = CoordinateSystem.WGS84_RA * (1 - CoordinateSystem.WGS84_E1) / (den_sqrt ** 3)

        sin2 = sin_lat * sin_lat
        sin4 = sin2 * sin2
        gamma_a = 9.7803267715
        gamma_0 = gamma_a * (
                    1 + 0.0052790414 * sin2 + 0.0000232718 * sin4 + 0.0000001262 * sin2 * sin4 + 0.0000000007 * sin4 * sin4)
        gamma = gamma_0 - (3.0877e-6 - 4.3e-9 * sin2) * h + 0.72e-12 * h * h
        return rm, rn, gamma

    @staticmethod
    def euler_to_matrix(euler: np.ndarray) -> np.ndarray:
        y, p, r = euler[2], euler[1], euler[0]
        cy, sy = np.cos(y), np.sin(y)
        cp, sp = np.cos(p), np.sin(p)
        cr, sr = np.cos(r), np.sin(r)

        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])

        return Rz @ Ry @ Rx

    @staticmethod
    def matrix_to_euler(dcm: np.ndarray) -> np.ndarray:
        euler = np.zeros(3)
        euler[1] = np.arcsin(-dcm[2, 0])
        if np.abs(np.cos(euler[1])) > 1e-10:
            euler[0] = np.arctan2(dcm[2, 1], dcm[2, 2])
            euler[2] = np.arctan2(dcm[1, 0], dcm[0, 0])
        else:
            euler[0] = 0.0
            euler[2] = np.arctan2(-dcm[0, 1], dcm[1, 1])
        return euler

    @staticmethod
    def rot_vec_to_quaternion(rot_vec: np.ndarray) -> np.ndarray:
        angle = np.linalg.norm(rot_vec)
        if angle < 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0])
        axis = rot_vec / angle
        half_angle = angle / 2.0
        w = np.cos(half_angle)
        v = axis * np.sin(half_angle)
        return np.array([w, v[0], v[1], v[2]])

    @staticmethod
    def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        res = np.array([w, x, y, z])
        return res / np.linalg.norm(res)

    @staticmethod
    def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
        w, x, y, z = q / np.linalg.norm(q)
        return np.array([
            [1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x ** 2 - 2 * y ** 2]
        ])

    @staticmethod
    def matrix_to_quaternion(dcm: np.ndarray) -> np.ndarray:
        tr = np.trace(dcm)
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            w = 0.25 * S
            x = (dcm[2, 1] - dcm[1, 2]) / S
            y = (dcm[0, 2] - dcm[2, 0]) / S
            z = (dcm[1, 0] - dcm[0, 1]) / S
        elif (dcm[0, 0] > dcm[1, 1]) and (dcm[0, 0] > dcm[2, 2]):
            S = np.sqrt(1.0 + dcm[0, 0] - dcm[1, 1] - dcm[2, 2]) * 2
            w = (dcm[2, 1] - dcm[1, 2]) / S
            x = 0.25 * S
            y = (dcm[0, 1] + dcm[1, 0]) / S
            z = (dcm[0, 2] + dcm[2, 0]) / S
        elif dcm[1, 1] > dcm[2, 2]:
            S = np.sqrt(1.0 + dcm[1, 1] - dcm[0, 0] - dcm[2, 2]) * 2
            w = (dcm[0, 2] - dcm[2, 0]) / S
            x = (dcm[0, 1] + dcm[1, 0]) / S
            y = 0.25 * S
            z = (dcm[1, 2] + dcm[2, 1]) / S
        else:
            S = np.sqrt(1.0 + dcm[2, 2] - dcm[0, 0] - dcm[1, 1]) * 2
            w = (dcm[1, 0] - dcm[0, 1]) / S
            x = (dcm[0, 2] + dcm[2, 0]) / S
            y = (dcm[1, 2] + dcm[2, 1]) / S
            z = 0.25 * S
        return np.array([w, x, y, z])

    @staticmethod
    def skew_symmetric(vec: np.ndarray) -> np.ndarray:
        return np.array([[0, -vec[2], vec[1]], [vec[2], 0, -vec[0]], [-vec[1], vec[0], 0]])

    @staticmethod
    def DRi(pos: np.ndarray) -> np.ndarray:
        rm, rn, _ = CoordinateSystem.get_radii_and_gravity(pos)
        lat, h = pos[0], pos[2]
        return np.diag([1.0 / (rm + h), 1.0 / ((rn + h) * np.cos(lat)), -1.0])

    @staticmethod
    def DR(pos: np.ndarray) -> np.ndarray:
        rm, rn, _ = CoordinateSystem.get_radii_and_gravity(pos)
        lat, h = pos[0], pos[2]
        return np.diag([(rm + h), (rn + h) * np.cos(lat), -1.0])