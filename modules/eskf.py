# multi_source_fusion/modules/eskf.py

import numpy as np
from typing import Dict, Optional, Tuple
from enum import Enum
import logging

try:
    from core.types import SystemState
    from core.config import config
    from core.coordinates import CoordinateSystem
except ImportError:
    import sys, os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.types import SystemState
    from core.config import config
    from core.coordinates import CoordinateSystem


class FusionMode(Enum):
    """融合模式枚举"""
    MATH_ONLY = "math_only"
    MODEL_ONLY = "model_only"
    WEIGHTED_FUSION = "weighted_fusion"
    ADAPTIVE_FUSION = "adaptive_fusion"


class ErrorStateKalmanFilter:
    """
    Error State Kalman Filter (ESKF) for GNSS/INS Integration
    基于KF-GINS的标准实现，使用21维误差状态向量

    状态向量顺序（与KF-GINS保持一致）：
    - 位置误差 (3): δr_n = [δlat, δlon, δh]           (索引 0-2)
    - 速度误差 (3): δv_n = [δv_e, δv_n, δv_u]        (索引 3-5)
    - 姿态误差 (3): φ_n = [φ_e, φ_n, φ_u]            (索引 6-8)
    - 陀螺仪偏置 (3): δb_g = [δb_gx, δb_gy, δb_gz]    (索引 9-11)
    - 加速度计偏置 (3): δb_a = [δb_ax, δb_ay, δb_az]  (索引 12-14)
    - 陀螺仪比例因子 (3): δs_g = [δs_gx, δs_gy, δs_gz] (索引 15-17)
    - 加速度计比例因子 (3): δs_a = [δs_ax, δs_ay, δs_az] (索引 18-20)
    """

    def __init__(self, mtinn_predictor, initial_state: SystemState):
        self.predictor = mtinn_predictor

        # 设置日志
        self.logger = logging.getLogger(__name__)

        # 融合模式
        self.fusion_mode, self.model_weight = config.validate_fusion_strategy()
        self.fusion_mode = FusionMode(self.fusion_mode)

        # 物理常数
        self.gravity = 9.80665  # m/s²
        self.earth_rate = CoordinateSystem.EARTH_RATE  # rad/s

        # === 名义状态 (Nominal State) ===
        self.pos_n = initial_state.position.copy()  # [lat, lon, h] (rad, rad, m)
        self.vel_n = initial_state.velocity.copy()  # [v_e, v_n, v_u] (m/s)
        self.att_n = initial_state.attitude.copy()  # [roll, pitch, yaw] (rad)

        # IMU偏置估计（初始化为零）
        self.bias_gyro = np.zeros(3)  # 陀螺仪偏置 (rad/s)
        self.bias_accel = np.zeros(3)  # 加速度计偏置 (m/s²)

        # IMU比例因子估计（初始化为1）
        self.scale_gyro = np.ones(3)  # 陀螺仪比例因子
        self.scale_accel = np.ones(3)  # 加速度计比例因子

        # === 误差状态协方差矩阵 (21x21) ===
        self.P = self._initialize_covariance()

        # === 过程噪声和测量噪声 ===
        self.Q = self._initialize_process_noise()
        self.R_gnss = self._initialize_measurement_noise()

        # === 时间相关变量 ===
        self.timestamp = initial_state.timestamp
        self.dt = 0.0

        # === 上一时刻的IMU数据（用于积分） ===
        self.last_gyro = np.zeros(3)
        self.last_accel = np.zeros(3)

        # === 导航参数缓存 ===
        self._last_nav_params = None

        self.logger.info("ESKF初始化完成")
        self.logger.info(f"融合模式: {self.fusion_mode.value}")
        self.logger.info(
            f"初始位置: lat={np.degrees(self.pos_n[0]):.6f}°, lon={np.degrees(self.pos_n[1]):.6f}°, h={self.pos_n[2]:.2f}m")

    def _initialize_covariance(self) -> np.ndarray:
        """初始化误差状态协方差矩阵 - 参考KF-GINS"""
        P = np.zeros((21, 21))

        # 位置误差初始不确定性 (rad², rad², m²)
        pos_std = config.get('initial_state.init_pos_std', [2.0, 2.0, 5.0])
        # 纬度和经度转换为弧度的标准差
        lat_std_rad = pos_std[0] / (6378137.0 * np.cos(self.pos_n[0]))  # 米转弧度
        lon_std_rad = pos_std[1] / 6378137.0  # 米转弧度
        P[0:3, 0:3] = np.diag([lat_std_rad ** 2, lon_std_rad ** 2, pos_std[2] ** 2])

        # 速度误差初始不确定性 (m²/s²)
        vel_std = config.get('initial_state.init_vel_std', [0.1, 0.1, 0.1])
        P[3:6, 3:6] = np.diag([std ** 2 for std in vel_std])

        # 姿态误差初始不确定性 (rad²)
        att_std_deg = config.get('initial_state.init_att_std', [0.1, 0.1, 0.5])
        att_std_rad = [np.radians(std) for std in att_std_deg]
        P[6:9, 6:9] = np.diag([std ** 2 for std in att_std_rad])

        # 陀螺仪偏置初始不确定性 (rad²/s²)
        gyro_bias_std = config.get('sensor_params.imu.gyro_bias_std', 1.0e-4)
        P[9:12, 9:12] = np.diag([gyro_bias_std ** 2] * 3)

        # 加速度计偏置初始不确定性 (m²/s⁴)
        accel_bias_std = config.get('sensor_params.imu.accel_bias_std', 1.0e-3)
        P[12:15, 12:15] = np.diag([accel_bias_std ** 2] * 3)

        # 陀螺仪比例因子初始不确定性
        gyro_scale_std = config.get('sensor_params.imu.gyro_scale_std', 1.0e-3)
        P[15:18, 15:18] = np.diag([gyro_scale_std ** 2] * 3)

        # 加速度计比例因子初始不确定性
        accel_scale_std = config.get('sensor_params.imu.accel_scale_std', 1.0e-3)
        P[18:21, 18:21] = np.diag([accel_scale_std ** 2] * 3)

        return P

    def _initialize_process_noise(self) -> np.ndarray:
        """初始化过程噪声协方差矩阵 - 参考KF-GINS标准设置"""
        Q = np.zeros((21, 21))

        # 位置误差过程噪声 (通常很小，因为位置由速度积分得到)
        Q[0:3, 0:3] = np.diag([1e-10, 1e-10, 1e-10])

        # 速度误差过程噪声 (由加速度计噪声驱动)
        accel_noise_std = config.get('sensor_params.imu.accel_noise_std', 1.0e-3)
        Q[3:6, 3:6] = np.diag([accel_noise_std ** 2] * 3)

        # 姿态误差过程噪声 (由陀螺仪噪声驱动)
        gyro_noise_std = config.get('sensor_params.imu.gyro_noise_std', 1.0e-4)
        Q[6:9, 6:9] = np.diag([gyro_noise_std ** 2] * 3)

        # 陀螺仪偏置随机游走
        gyro_bias_walk = config.get('sensor_params.imu.gyro_bias_walk', 1.0e-6)
        Q[9:12, 9:12] = np.diag([gyro_bias_walk ** 2] * 3)

        # 加速度计偏置随机游走
        accel_bias_walk = config.get('sensor_params.imu.accel_bias_walk', 1.0e-5)
        Q[12:15, 12:15] = np.diag([accel_bias_walk ** 2] * 3)

        # 比例因子随机游走
        scale_walk = config.get('sensor_params.imu.scale_walk', 1.0e-8)
        Q[15:18, 15:18] = np.diag([scale_walk ** 2] * 3)  # 陀螺仪比例因子
        Q[18:21, 18:21] = np.diag([scale_walk ** 2] * 3)  # 加速度计比例因子

        return Q

    def _initialize_measurement_noise(self) -> np.ndarray:
        """初始化GNSS测量噪声协方差矩阵"""
        pos_std_h = config.get('sensor_params.gnss.pos_std_h', 1.0)
        pos_std_v = config.get('sensor_params.gnss.pos_std_v', 2.0)
        return np.diag([pos_std_h ** 2, pos_std_h ** 2, pos_std_v ** 2])

    # multi_source_fusion/modules/eskf.py 中的修改

    def predict(self, imu_data: Dict[str, float], odo_data: Optional[Dict[str, float]], dt: float,
                feedback_error: Optional[np.ndarray] = None):
        """
        ESKF预测步骤 - 基于KF-GINS的标准实现
        """
        if dt <= 0 or dt > 1.0:
            self.logger.warning(f"异常的时间间隔 dt={dt:.6f}s, 跳过预测")
            return

        self.dt = dt

        # 读取IMU数据
        gyro_raw = np.array([imu_data['gx'], imu_data['gy'], imu_data['gz']])
        accel_raw = np.array([imu_data['ax'], imu_data['ay'], imu_data['az']])

        # IMU误差补偿
        gyro_corrected = self._compensate_gyro(gyro_raw)
        accel_corrected = self._compensate_accel(accel_raw)

        # 1. 名义状态传播 (Nominal State Propagation) - 严格的INS机械编排
        self._propagate_nominal_state_strict(gyro_corrected, accel_corrected, dt)

        # 2. 误差状态传播 (Error State Propagation)
        self._propagate_error_state_kfgins(gyro_corrected, accel_corrected, dt)

        # 3. 里程计约束（如果有）
        if odo_data is not None:
            self._apply_odometer_constraint(odo_data)

        # 4. 神经网络校正（如果启用）
        if self.fusion_mode != FusionMode.MATH_ONLY and self.predictor is not None:
            # 为神经网络提供默认的odo_data
            odo_for_nn = odo_data if odo_data is not None else {'velocity': 0.0}
            self._apply_model_correction(imu_data, odo_for_nn, feedback_error)

        # 更新时间戳
        self.timestamp += dt

        # 保存当前IMU数据用于下次积分
        self.last_gyro = gyro_corrected.copy()
        self.last_accel = accel_corrected.copy()

    def _apply_odometer_constraint(self, odo_data: Dict[str, float]):
        """
        应用里程计约束 - 使用伪测量更新
        """
        try:
            # 里程计测量的是载体坐标系前向速度
            v_odo = odo_data['velocity']

            # 将载体坐标系速度转换为导航坐标系
            C_nb = self._euler_to_dcm(self.att_n)
            v_body = np.array([0, v_odo, 0])  # 假设前向为Y轴
            v_nav_pred = C_nb @ v_body

            # 构造观测矩阵（观测导航系速度的模长）
            v_nav_norm = np.linalg.norm(self.vel_n)
            if v_nav_norm > 1e-6:
                H_odo = np.zeros((1, 21))
                H_odo[0, 3:6] = self.vel_n / v_nav_norm  # 对速度的偏导数

                # 观测残差
                z_odo = np.array([v_odo])
                h_odo = np.array([v_nav_norm])
                y_odo = z_odo - h_odo

                # 观测噪声
                R_odo = np.array([[config.get('sensor_params.odo.vel_std', 0.05) ** 2]])

                # 卡尔曼更新
                S_odo = H_odo @ self.P @ H_odo.T + R_odo
                if np.linalg.cond(S_odo) < 1e12:
                    K_odo = self.P @ H_odo.T @ np.linalg.inv(S_odo)
                    delta_x_odo = K_odo @ y_odo

                    # 限制校正幅度
                    delta_x_odo = np.clip(delta_x_odo, -0.1, 0.1)

                    # 应用校正
                    self._inject_error_state(delta_x_odo)

                    # 协方差更新
                    I_KH = np.eye(21) - K_odo @ H_odo
                    self.P = I_KH @ self.P @ I_KH.T + K_odo @ R_odo @ K_odo.T
                    self.P = self._ensure_positive_definite(self.P)

        except Exception as e:
            self.logger.warning(f"里程计约束应用失败: {e}")

    def _compensate_gyro(self, gyro_raw: np.ndarray) -> np.ndarray:
        """陀螺仪误差补偿"""
        return (gyro_raw - self.bias_gyro) * self.scale_gyro

    def _compensate_accel(self, accel_raw: np.ndarray) -> np.ndarray:
        """加速度计误差补偿"""
        return (accel_raw - self.bias_accel) * self.scale_accel

    def _propagate_nominal_state_strict(self, gyro: np.ndarray, accel: np.ndarray, dt: float):
        """
        严格的名义状态传播 - 基于KF-GINS的INS机械编排
        使用中点积分和二阶修正
        """
        # 当前位置参数
        lat, lon, h = self.pos_n[0], self.pos_n[1], self.pos_n[2]

        # 计算地球曲率半径
        R_M, R_N = CoordinateSystem.get_radii(lat)

        # === 1. 姿态更新 ===
        # 计算导航系相对地球的角速度
        omega_en_n = np.array([
            self.vel_n[0] / (R_N + h),  # 东向速度引起的角速度
            -self.vel_n[1] / (R_M + h),  # 北向速度引起的角速度
            -self.vel_n[0] * np.tan(lat) / (R_N + h)  # 东向速度在纬度方向的投影
        ])

        # 地球自转角速度在导航系的投影
        omega_ie_n = np.array([
            self.earth_rate * np.cos(lat),
            0,
            -self.earth_rate * np.sin(lat)
        ])

        # 导航系相对惯性系的角速度
        omega_in_n = omega_ie_n + omega_en_n

        # 载体系相对导航系的角速度（补偿后的陀螺仪输出）
        C_nb = self._euler_to_dcm(self.att_n)
        omega_nb_n = C_nb @ gyro

        # 载体系相对惯性系的角速度
        omega_ib_n = omega_in_n + omega_nb_n

        # 使用四元数进行姿态更新（避免奇点）
        q_nb = self._dcm_to_quaternion(C_nb)
        q_nb_new = self._update_quaternion_midpoint(q_nb, omega_ib_n, dt)
        C_nb_new = self._quaternion_to_dcm(q_nb_new)
        self.att_n = self._dcm_to_euler(C_nb_new)

        # === 2. 速度更新 ===
        # 比力转换到导航系（使用中点方法）
        C_nb_mid = self._quaternion_to_dcm(self._slerp_quaternion(q_nb, q_nb_new, 0.5))
        f_n = C_nb_mid @ accel

        # 重力在导航系的表示（考虑高度变化）
        g_n = np.array([0, 0, -self._gravity_model(lat, h)])

        # Coriolis力和向心力
        coriolis_centripetal = -np.cross(2 * omega_ie_n + omega_en_n, self.vel_n)

        # 速度微分方程
        vel_dot = f_n + g_n + coriolis_centripetal

        # 使用梯形积分更新速度
        self.vel_n += vel_dot * dt

        # === 3. 位置更新 ===
        # 使用中点速度进行位置更新
        vel_mid = self.vel_n - 0.5 * vel_dot * dt

        # 更新曲率半径（使用中点位置）
        lat_mid = lat + 0.5 * vel_mid[1] * dt / (R_M + h)
        R_M_mid, R_N_mid = CoordinateSystem.get_radii(lat_mid)
        h_mid = h + 0.5 * vel_mid[2] * dt

        # 位置微分方程
        pos_dot = np.array([
            vel_mid[1] / (R_M_mid + h_mid),  # 纬度变化率
            vel_mid[0] / ((R_N_mid + h_mid) * np.cos(lat_mid)),  # 经度变化率
            vel_mid[2]  # 高程变化率
        ])

        self.pos_n += pos_dot * dt

        # 角度环绕处理
        self.att_n = self._wrap_angles(self.att_n)
        self.pos_n[1] = self._wrap_to_pi(self.pos_n[1])  # 经度环绕

    def _propagate_error_state_kfgins(self, gyro: np.ndarray, accel: np.ndarray, dt: float):
        """
        误差状态传播 - 基于KF-GINS的标准实现
        """
        # 计算状态转移矩阵 F (21x21)
        F = self._compute_state_transition_matrix_kfgins(gyro, accel)

        # 离散化状态转移矩阵 - 使用一阶近似
        Phi = np.eye(21) + F * dt + 0.5 * (F @ F) * dt ** 2

        # 过程噪声的离散化
        Q_d = self.Q * dt

        # 协方差传播
        self.P = Phi @ self.P @ Phi.T + Q_d

        # 确保协方差矩阵正定
        self.P = self._ensure_positive_definite(self.P)


    def _compute_state_transition_matrix_kfgins(self, gyro: np.ndarray, accel: np.ndarray) -> np.ndarray:
        """
        计算21x21状态转移矩阵 - 修复数组越界问题
        """
        F = np.zeros((21, 21))

        lat, lon, h = self.pos_n[0], self.pos_n[1], self.pos_n[2]
        R_M, R_N = CoordinateSystem.get_radii(lat)

        # 确保分母不为零
        R_M_h = R_M + h
        R_N_h = R_N + h
        cos_lat = np.cos(lat)
        sin_lat = np.sin(lat)
        tan_lat = np.tan(lat)

        # 地球自转角速度
        omega_ie = self.earth_rate

        # === F_rr (位置对位置的偏导数) ===
        F[0, 2] = -self.vel_n[1] / (R_M_h ** 2)  # ∂δlat/∂δh
        F[1, 0] = self.vel_n[0] * tan_lat / (R_N_h * cos_lat)  # ∂δlon/∂δlat
        F[1, 2] = -self.vel_n[0] / (R_N_h ** 2 * cos_lat)  # ∂δlon/∂δh

        # === F_rv (位置对速度的偏导数) ===
        F[0, 4] = 1 / R_M_h  # ∂δlat/∂δv_n
        F[1, 3] = 1 / (R_N_h * cos_lat)  # ∂δlon/∂δv_e
        F[2, 5] = 1  # ∂δh/∂δv_u

        # === F_vr (速度对位置的偏导数) ===
        # 修复：这里之前错误地使用了self.vel_n[4]和self.vel_n[5]，但vel_n只有3个元素
        F[3, 0] = -2 * omega_ie * self.vel_n[2] * sin_lat
        F[3, 2] = self.vel_n[1] ** 2 / (R_M_h ** 2) - self.vel_n[0] ** 2 / (R_N_h ** 2)

        F[4, 0] = 2 * omega_ie * (self.vel_n[2] * cos_lat + self.vel_n[0] * sin_lat)
        F[4, 2] = -self.vel_n[0] * self.vel_n[1] / (R_M_h ** 2) - self.vel_n[0] * self.vel_n[1] / (R_N_h ** 2)

        F[5, 0] = -2 * omega_ie * self.vel_n[1] * cos_lat
        F[5, 2] = (self.vel_n[0] ** 2 / (R_N_h ** 2) + self.vel_n[1] ** 2 / (R_M_h ** 2))

        # === F_vv (速度对速度的偏导数) ===
        F[3, 3] = self.vel_n[2] / R_N_h
        F[3, 4] = -self.vel_n[2] / R_M_h
        F[3, 5] = self.vel_n[1] / R_M_h - self.vel_n[0] / R_N_h

        F[4, 3] = self.vel_n[2] / R_N_h + 2 * omega_ie * sin_lat
        F[4, 4] = self.vel_n[0] * tan_lat / R_N_h
        F[4, 5] = -(self.vel_n[0] / R_N_h + 2 * omega_ie * cos_lat)

        F[5, 3] = -2 * self.vel_n[1] / R_N_h
        F[5, 4] = 2 * self.vel_n[0] / R_M_h + 2 * omega_ie * cos_lat

        # === F_vφ (速度对姿态的偏导数) ===
        C_nb = self._euler_to_dcm(self.att_n)
        f_n = C_nb @ accel
        f_skew = self._skew_symmetric(f_n)
        F[3:6, 6:9] = f_skew

        # === F_φr (姿态对位置的偏导数) ===
        F[6, 0] = -omega_ie * sin_lat
        F[8, 0] = -omega_ie * cos_lat

        # === F_φv (姿态对速度的偏导数) ===
        F[6, 3] = 1 / R_N_h
        F[7, 4] = -1 / R_M_h
        F[8, 3] = -tan_lat / R_N_h

        # === F_φbg (姿态对陀螺仪偏置的偏导数) ===
        F[6:9, 9:12] = -C_nb

        # === F_vba (速度对加速度计偏置的偏导数) ===
        F[3:6, 12:15] = -C_nb

        # === F_φsg (姿态对陀螺仪比例因子的偏导数) ===
        gyro_diag = np.diag(gyro)
        F[6:9, 15:18] = -C_nb @ gyro_diag

        # === F_vsa (速度对加速度计比例因子的偏导数) ===
        accel_diag = np.diag(accel)
        F[3:6, 18:21] = -C_nb @ accel_diag

        return F
    def _gravity_model(self, lat: float, h: float) -> float:
        """
        重力模型 - 考虑纬度和高度的影响
        """
        # WGS84重力模型
        g0 = 9.7803267714  # 赤道重力
        g1 = 0.0052790414  # 重力扁率
        g2 = 0.0000232718  # 二阶项

        sin2_lat = np.sin(lat) ** 2
        g_lat = g0 * (1 + g1 * sin2_lat + g2 * np.sin(2 * lat) ** 2)

        # 高度修正
        g_h = g_lat * (1 - 2 * h / 6378137.0)

        return g_h

    def _apply_model_correction(self, imu_data: Dict[str, float], odo_data: Dict[str, float],
                                feedback_error: Optional[np.ndarray]):
        """应用神经网络模型校正"""
        if self.predictor is None:
            return

        try:
            # 获取模型预测的状态变化量
            model_delta_dict, _ = self.predictor.predict_step_with_uncertainty(
                imu_data, odo_data, self._get_current_system_state(), feedback_error
            )

            # 根据融合模式应用校正
            if self.fusion_mode == FusionMode.WEIGHTED_FUSION:
                alpha = self.model_weight

                # 限制模型校正幅度
                delta_att = np.clip(model_delta_dict['attitude'], -0.01, 0.01)
                delta_vel = np.clip(model_delta_dict['velocity'], -0.1, 0.1)
                delta_pos = np.clip(model_delta_dict['position'], -1e-5, 1e-5)

                # 应用校正到名义状态
                self.att_n += alpha * delta_att
                self.vel_n += alpha * delta_vel
                self.pos_n += alpha * delta_pos

                # 角度环绕处理
                self.att_n = self._wrap_angles(self.att_n)
                self.pos_n[1] = self._wrap_to_pi(self.pos_n[1])

        except Exception as e:
            self.logger.warning(f"模型校正失败: {e}")

    def update(self, gnss_data: Dict[str, float]):
        """ESKF更新步骤 - GNSS位置观测"""
        try:
            # 观测值 (lat, lon, h)
            z = np.array([gnss_data['lat'], gnss_data['lon'], gnss_data['h']])

            # 观测噪声协方差
            R = np.diag([
                gnss_data.get('std_lat', 1.0) ** 2,
                gnss_data.get('std_lon', 1.0) ** 2,
                gnss_data.get('std_h', 2.0) ** 2
            ])

            # 观测矩阵 H (3x21) - 只观测位置
            H = np.zeros((3, 21))
            H[0:3, 0:3] = np.eye(3)

            # 预测观测值
            h_x = self.pos_n.copy()

            # 观测残差
            y = z - h_x
            y[1] = self._wrap_to_pi(y[1])  # 经度残差环绕

            # 卡尔曼增益
            S = H @ self.P @ H.T + R

            # 检查可观测性
            if np.linalg.cond(S) > 1e12:
                self.logger.warning(f"观测矩阵条件数过大: {np.linalg.cond(S):.2e}")
                return

            K = self.P @ H.T @ np.linalg.inv(S)

            # 误差状态更新
            delta_x = K @ y

            # 限制校正幅度
            delta_x[0:3] = np.clip(delta_x[0:3], -0.001, 0.001)  # 位置校正
            delta_x[3:6] = np.clip(delta_x[3:6], -1.0, 1.0)  # 速度校正
            delta_x[6:9] = np.clip(delta_x[6:9], -0.1, 0.1)  # 姿态校正

            # 误差状态反馈到名义状态
            self._inject_error_state(delta_x)

            # 协方差更新 (Joseph形式)
            I_KH = np.eye(21) - K @ H
            self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
            self.P = self._ensure_positive_definite(self.P)

            # 打印更新信息
            residual_norm = np.linalg.norm(y)
            if residual_norm > 1e-6:
                self.logger.info(f"GNSS更新: 残差范数={residual_norm:.6f}")

        except Exception as e:
            self.logger.error(f"GNSS更新失败: {e}")

    def _inject_error_state(self, delta_x: np.ndarray):
        """将误差状态反馈到名义状态"""

        # 位置校正
        self.pos_n += delta_x[0:3]

        # 速度校正
        self.vel_n += delta_x[3:6]

        # 姿态校正 (phi-angle model)
        phi = delta_x[6:9]
        if np.linalg.norm(phi) > 1e-8:
            # 小角度假设下的旋转矩阵
            delta_C = np.eye(3) + self._skew_symmetric(phi)

            # 当前姿态的旋转矩阵
            C_nb = self._euler_to_dcm(self.att_n)

            # 更新后的旋转矩阵
            C_nb_new = delta_C @ C_nb

            # 转换回欧拉角
            self.att_n = self._dcm_to_euler(C_nb_new)

        # IMU偏置校正
        self.bias_gyro += delta_x[9:12]
        self.bias_accel += delta_x[12:15]

        # IMU比例因子校正
        self.scale_gyro += delta_x[15:18]
        self.scale_accel += delta_x[18:21]

        # 角度环绕处理
        self.att_n = self._wrap_angles(self.att_n)
        self.pos_n[1] = self._wrap_to_pi(self.pos_n[1])

    def get_state(self) -> SystemState:
        """获取当前系统状态"""
        # 从21x21协方差矩阵中提取主要状态的协方差
        # ESKF状态顺序：[pos(0-2), vel(3-5), att(6-8), ...]
        # SystemState期望顺序：[att, vel, pos]

        # 重新排列协方差矩阵以匹配SystemState的状态顺序
        eskf_indices = [6, 7, 8, 3, 4, 5, 0, 1, 2]  # att(6-8), vel(3-5), pos(0-2)
        main_covariance = self.P[np.ix_(eskf_indices, eskf_indices)]

        return SystemState(
            timestamp=self.timestamp,
            attitude=self.att_n.copy(),
            velocity=self.vel_n.copy(),
            position=self.pos_n.copy(),
            covariance=main_covariance
        )

    def _get_current_system_state(self) -> SystemState:
        """获取当前系统状态（用于模型预测）"""
        return self.get_state()

    # === 辅助函数 ===
    def _euler_to_dcm(self, euler: np.ndarray) -> np.ndarray:
        """欧拉角转方向余弦矩阵 - ZYX顺序"""
        roll, pitch, yaw = euler[0], euler[1], euler[2]

        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        # ZYX欧拉角的旋转矩阵
        C = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]
        ])

        return C

    def _dcm_to_euler(self, C: np.ndarray) -> np.ndarray:
        """方向余弦矩阵转欧拉角 - ZYX顺序"""
        roll = np.arctan2(C[2, 1], C[2, 2])
        pitch = np.arcsin(-np.clip(C[2, 0], -1, 1))
        yaw = np.arctan2(C[1, 0], C[0, 0])
        return np.array([roll, pitch, yaw])

    def _dcm_to_quaternion(self, C: np.ndarray) -> np.ndarray:
        """方向余弦矩阵转四元数"""
        trace = np.trace(C)

        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
            w = 0.25 * s
            x = (C[2, 1] - C[1, 2]) / s
            y = (C[0, 2] - C[2, 0]) / s
            z = (C[1, 0] - C[0, 1]) / s
        elif C[0, 0] > C[1, 1] and C[0, 0] > C[2, 2]:
            s = np.sqrt(1.0 + C[0, 0] - C[1, 1] - C[2, 2]) * 2
            w = (C[2, 1] - C[1, 2]) / s
            x = 0.25 * s
            y = (C[0, 1] + C[1, 0]) / s
            z = (C[0, 2] + C[2, 0]) / s
        elif C[1, 1] > C[2, 2]:
            s = np.sqrt(1.0 + C[1, 1] - C[0, 0] - C[2, 2]) * 2
            w = (C[0, 2] - C[2, 0]) / s
            x = (C[0, 1] + C[1, 0]) / s
            y = 0.25 * s
            z = (C[1, 2] + C[2, 1]) / s
        else:
            s = np.sqrt(1.0 + C[2, 2] - C[0, 0] - C[1, 1]) * 2
            w = (C[1, 0] - C[0, 1]) / s
            x = (C[0, 2] + C[2, 0]) / s
            y = (C[1, 2] + C[2, 1]) / s
            z = 0.25 * s

        return np.array([w, x, y, z])

    def _quaternion_to_dcm(self, q: np.ndarray) -> np.ndarray:
        """四元数转方向余弦矩阵"""
        w, x, y, z = q[0], q[1], q[2], q[3]

        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]
        ])

    def _update_quaternion_midpoint(self, q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
        """四元数更新 - 使用中点积分方法"""
        omega_norm = np.linalg.norm(omega)

        if omega_norm < 1e-8:
            return q

        # 旋转向量
        axis = omega / omega_norm
        angle = omega_norm * dt

        # 增量四元数
        dq = np.array([
            np.cos(angle / 2),
            axis[0] * np.sin(angle / 2),
            axis[1] * np.sin(angle / 2),
            axis[2] * np.sin(angle / 2)
        ])

        # 四元数乘法
        q_new = self._quaternion_multiply(q, dq)
        return q_new / np.linalg.norm(q_new)

    def _slerp_quaternion(self, q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
        """球面线性插值"""
        dot = np.dot(q1, q2)

        if dot < 0:
            q2 = -q2
            dot = -dot

        if dot > 0.9995:
            # 线性插值
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)

        theta_0 = np.arccos(np.abs(dot))
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * t
        sin_theta = np.sin(theta)

        s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
        s1 = sin_theta / sin_theta_0

        return s0 * q1 + s1 * q2

    def _quaternion_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """四元数乘法"""
        w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
        w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]

        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        ])

    def _skew_symmetric(self, v: np.ndarray) -> np.ndarray:
        """构造反对称矩阵"""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    def _ensure_positive_definite(self, matrix: np.ndarray, min_eigenvalue: float = 1e-12) -> np.ndarray:
        """确保矩阵正定"""
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
            eigenvalues[eigenvalues < min_eigenvalue] = min_eigenvalue
            return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        except:
            return np.diag(np.abs(np.diag(matrix))) + min_eigenvalue * np.eye(matrix.shape[0])

    def _wrap_to_pi(self, angle: float) -> float:
        """角度环绕到[-π, π]"""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _wrap_angles(self, angles: np.ndarray) -> np.ndarray:
        """角度环绕处理"""
        angles = angles.copy()
        angles[0] = self._wrap_to_pi(angles[0])  # roll
        angles[1] = np.clip(angles[1], -np.pi / 2, np.pi / 2)  # pitch限制
        angles[2] = self._wrap_to_pi(angles[2])  # yaw
        return angles


# 为了保持与原代码的兼容性，创建一个别名
MTINN_EKF = ErrorStateKalmanFilter