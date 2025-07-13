# multi_source_fusion/modules/eskf.py

import numpy as np
from dataclasses import field
import logging

from core.coordinates import CoordinateSystem
from core.config import config
from core.types import PVA, Attitude, ImuError, IMU, GNSS, NavState


class _INSMech:
    """
    最终修正版: 完全、精确地仿照 insmech.cpp 中 velUpdate, posUpdate, attUpdate 的实现。
    此版本确保了数值稳定性和精度，是解决问题的关键。
    """

    @staticmethod
    def ins_mech(pvapre: PVA, imupre: IMU, imucur: IMU) -> PVA:
        dt = imucur.dt
        if dt <= 0:
            return pvapre

        pvacur = PVA(pos=pvapre.pos.copy(), vel=pvapre.vel.copy(), att=Attitude())

        # --- 第一步：速度更新 (velUpdate) ---
        # 1.1 计算 k-1/2 (中点) 时刻的地理参数
        rm_pre, rn_pre, g_pre = CoordinateSystem.get_radii_and_gravity(pvapre.pos)
        wie_n = np.array([CoordinateSystem.WGS84_WIE * np.cos(pvapre.pos[0]), 0,
                          -CoordinateSystem.WGS84_WIE * np.sin(pvapre.pos[0])])
        wen_n = np.array([
            pvapre.vel[1] / (rn_pre + pvapre.pos[2]),
            -pvapre.vel[0] / (rm_pre + pvapre.pos[2]),
            -pvapre.vel[1] * np.tan(pvapre.pos[0]) / (rn_pre + pvapre.pos[2])
        ])

        # 1.2 计算速度增量 (比力 + 重力/哥氏力)
        d_vfb = imucur.dvel + 0.5 * np.cross(imucur.dtheta, imucur.dvel) + \
                (np.cross(imupre.dtheta, imucur.dvel) + np.cross(imupre.dvel, imucur.dtheta)) / 12.0

        cnn = np.eye(3) - 0.5 * CoordinateSystem.skew_symmetric((wie_n + wen_n) * dt)
        d_vfn = cnn @ pvapre.att.cbn @ d_vfb

        gl = np.array([0, 0, g_pre])
        d_vgn = (gl - np.cross(2 * wie_n + wen_n, pvapre.vel)) * dt

        # 1.3 第一次近似更新速度
        pvacur.vel = pvapre.vel + d_vfn + d_vgn

        # --- 第二步：位置更新 (posUpdate) ---
        # 2.1 计算精确的中间时刻速度和位置
        mid_vel = 0.5 * (pvapre.vel + pvacur.vel)
        mid_pos = pvapre.pos + 0.5 * (CoordinateSystem.DRi(pvapre.pos) @ mid_vel) * dt

        # 2.2 使用中间位置更新位置
        pvacur.pos = pvapre.pos + (CoordinateSystem.DRi(mid_pos) @ mid_vel) * dt

        # --- 第三步：姿态更新 (attUpdate) ---
        # 3.1 重新计算中间时刻的地理参数
        rm_mid, rn_mid, _ = CoordinateSystem.get_radii_and_gravity(mid_pos)
        lat_mid, h_mid = mid_pos[0], mid_pos[2]

        wie_n_mid = np.array(
            [CoordinateSystem.WGS84_WIE * np.cos(lat_mid), 0, -CoordinateSystem.WGS84_WIE * np.sin(lat_mid)])
        wen_n_mid = np.array([
            mid_vel[1] / (rn_mid + h_mid),
            -mid_vel[0] / (rm_mid + h_mid),
            -mid_vel[1] * np.tan(lat_mid) / (rn_mid + h_mid)
        ])

        # 3.2 计算 n 系和 b 系的旋转四元数
        alpha_nin = -(wie_n_mid + wen_n_mid) * dt
        q_nn = CoordinateSystem.rot_vec_to_quaternion(alpha_nin)

        beta_bib = imucur.dtheta + np.cross(imupre.dtheta, imucur.dtheta) / 12.0
        q_bb = CoordinateSystem.rot_vec_to_quaternion(beta_bib)

        # 3.3 更新姿态
        pvacur.att.qbn = CoordinateSystem.quaternion_multiply(q_nn, pvapre.att.qbn)
        pvacur.att.qbn = CoordinateSystem.quaternion_multiply(pvacur.att.qbn, q_bb)

        # 3.4 更新 Cbn 和欧拉角
        pvacur.att.cbn = CoordinateSystem.quaternion_to_matrix(pvacur.att.qbn)
        pvacur.att.euler = CoordinateSystem.matrix_to_euler(pvacur.att.cbn)

        return pvacur


class ErrorStateKalmanFilter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._load_options()

        self.pvacur = PVA(
            pos=np.radians(np.array(self.options['init_state']['pos'])),
            vel=np.array(self.options['init_state']['vel']),
            att=Attitude(euler=np.radians(np.array(self.options['init_state']['att'])))
        )
        self.pvacur.pos[2] = self.options['init_state']['pos'][2]
        self.pvacur.att.cbn = CoordinateSystem.euler_to_matrix(self.pvacur.att.euler)
        self.pvacur.att.qbn = CoordinateSystem.matrix_to_quaternion(self.pvacur.att.cbn)
        self.pvapre = self.pvacur

        self.imuerror = ImuError(
            gyrbias=np.radians(np.array(self.options['init_state']['gyrbias'])) / 3600.0,
            accbias=np.array(self.options['init_state']['accbias']) * 1e-5,
            gyrscale=np.array(self.options['init_state']['gyrscale']) * 1e-6,
            accscale=np.array(self.options['init_state']['accscale']) * 1e-6
        )

        self.P = self._initialize_covariance()
        self.Qc = self._initialize_process_noise()
        self.dx = np.zeros(21)

        self.imupre = IMU(time=0.0)
        self.imucur = self.imupre
        self.gnssdata = None

    def add_imu_data(self, imu: IMU):
        # imucur 在 main.py 中被赋值
        self.imucur = imu

    def add_gnss_data(self, gnss: GNSS):
        self.gnssdata = gnss

    def new_imu_process(self):
        compensated_imucur = self._imu_compensate(self.imucur)
        self._ins_propagation(self.imupre, compensated_imucur)
        self.pvapre = self.pvacur
        self.imupre = compensated_imucur

    def _ins_propagation(self, imupre: IMU, imucur: IMU):
        self.pvacur = _INSMech.ins_mech(self.pvapre, imupre, imucur)

        if not np.all(np.isfinite(self.pvacur.pos)) or not np.all(np.isfinite(self.pvacur.vel)) or not np.all(
                np.isfinite(self.pvacur.att.euler)):
            self.logger.error(f"NaN or Inf detected in PVA state at time {imucur.time}. Halting propagation.")
            raise ValueError(f"PVA state became invalid (NaN or Inf) at time {imucur.time}")

        F = self._compute_F_matrix(self.pvapre, imucur)
        G = self._compute_G_matrix(self.pvapre)

        dt = imucur.dt
        if dt <= 0: return

        Phi = np.eye(21) + F * dt
        Qd = (Phi @ G @ self.Qc @ G.T @ Phi.T + G @ self.Qc @ G.T) * dt / 2.0

        self.P = Phi @ self.P @ Phi.T + Qd
        self.dx = Phi @ self.dx
        self._check_cov()

    def gnss_update(self):
        ant_lever = np.array(self.options['ant_lever'])
        pos_ant_body = self.pvacur.att.cbn @ ant_lever
        pos_ant_calculated = self.pvacur.pos + CoordinateSystem.DRi(self.pvacur.pos) @ pos_ant_body

        # 新息 dz = calc - meas (与 kf-gins 一致)
        dz = CoordinateSystem.DR(self.pvacur.pos) @ (pos_ant_calculated - self.gnssdata.pos)

        H = np.zeros((3, 21))
        # H 矩阵 (与 kf-gins 一致)
        H[:, 0:3] = np.eye(3)
        H[:, 6:9] = -CoordinateSystem.skew_symmetric(pos_ant_body)

        R = np.diag(self.gnssdata.std ** 2)

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        # 更新 (与 kf-gins 一致)
        self.dx = self.dx + K @ (dz - H @ self.dx)
        I = np.eye(21) - K @ H
        self.P = I @ self.P @ I.T + K @ R @ K.T
        self._check_cov()

        self._state_feedback()
        self.gnssdata = None

    def _state_feedback(self):
        delta_r = self.dx[0:3]
        self.pvacur.pos -= CoordinateSystem.DRi(self.pvacur.pos) @ delta_r
        self.pvacur.vel -= self.dx[3:6]
        rot_vec = self.dx[6:9]
        q_pn = CoordinateSystem.rot_vec_to_quaternion(rot_vec)
        self.pvacur.att.qbn = CoordinateSystem.quaternion_multiply(q_pn, self.pvacur.att.qbn)
        self.pvacur.att.cbn = CoordinateSystem.quaternion_to_matrix(self.pvacur.att.qbn)
        self.pvacur.att.euler = CoordinateSystem.matrix_to_euler(self.pvacur.att.cbn)
        self.imuerror.gyrbias += self.dx[9:12]
        self.imuerror.accbias += self.dx[12:15]
        self.imuerror.gyrscale += self.dx[15:18]
        self.imuerror.accscale += self.dx[18:21]
        self.dx.fill(0)

    def _compute_F_matrix(self, pva: PVA, imu: IMU) -> np.ndarray:
        F = np.zeros((21, 21))
        lat, h = pva.pos[0], pva.pos[2]
        v_n, v_e, v_d = pva.vel[0], pva.vel[1], pva.vel[2]

        if imu.dt <= 0: return F

        accel = imu.dvel / imu.dt
        omega = imu.dtheta / imu.dt

        rm, rn, g = CoordinateSystem.get_radii_and_gravity(pva.pos)
        rmh, rnh = rm + h, rn + h

        wie = CoordinateSystem.WGS84_WIE
        sin_lat, cos_lat, tan_lat = np.sin(lat), np.cos(lat), np.tan(lat)

        wie_n = np.array([wie * cos_lat, 0, -wie * sin_lat])
        wen_n = np.array([v_e / rnh, -v_n / rmh, -v_e * tan_lat / rnh])

        # F_pp
        F_pp = np.zeros((3, 3))
        F_pp[0, 0] = -v_d / rmh
        F_pp[0, 2] = v_n / rmh
        F_pp[1, 0] = v_e * tan_lat / rnh
        F_pp[1, 1] = -(v_d + v_n * tan_lat) / rnh
        F_pp[1, 2] = v_e / rnh
        F[0:3, 0:3] = F_pp

        # F_pv
        F[0:3, 3:6] = np.diag([1 / rmh, 1 / (rnh * cos_lat), -1])

        # F_vp
        F_vp = np.zeros((3, 3))
        F_vp[0, 0] = -2 * v_e * wie * cos_lat / rmh - v_e ** 2 / (rmh * rnh * cos_lat ** 2)
        F_vp[0, 2] = v_n * v_d / rmh ** 2 - v_e ** 2 * tan_lat / rnh ** 2
        F_vp[1, 0] = 2 * wie * (v_n * cos_lat - v_d * sin_lat) / rmh + v_n * v_e / (rmh * rnh * cos_lat ** 2)
        F_vp[1, 2] = (v_n * v_e * tan_lat + v_e * v_d) / rnh ** 2
        F_vp[2, 0] = 2 * wie * v_e * sin_lat / rmh
        F_vp[2, 2] = -(v_e ** 2 / rnh ** 2 + v_n ** 2 / rmh ** 2) + 2 * g / (np.sqrt(rm * rn) + h)
        F[3:6, 0:3] = F_vp

        # F_vv (This was the main source of error)
        F_vv = np.zeros((3, 3))
        F_vv[0, 0] = v_d / rmh
        F_vv[0, 1] = -2 * (wie * sin_lat + v_e * tan_lat / rnh)
        F_vv[0, 2] = v_n / rmh
        F_vv[1, 0] = 2 * wie * sin_lat + v_e * tan_lat / rnh
        F_vv[1, 1] = (v_d + v_n * tan_lat) / rnh
        F_vv[1, 2] = 2 * wie * cos_lat + v_e / rnh
        F_vv[2, 0] = -2 * v_n / rmh
        F_vv[2, 1] = -2 * (wie * cos_lat + v_e / rnh)
        F[3:6, 3:6] = F_vv

        # F_ve, F_va, F_vs
        F[3:6, 6:9] = CoordinateSystem.skew_symmetric(pva.att.cbn @ accel)
        F[3:6, 12:15] = pva.att.cbn
        F[3:6, 18:21] = pva.att.cbn @ np.diag(accel)

        # F_ap
        F_ap = np.zeros((3, 3))
        F_ap[0, 0] = -wie * sin_lat / rmh
        F_ap[0, 2] = v_e / rnh ** 2
        F_ap[1, 2] = -v_n / rmh ** 2
        F_ap[2, 0] = -wie * cos_lat / rmh - v_e / (rmh * rnh * cos_lat ** 2)
        F_ap[2, 2] = -v_e * tan_lat / rnh ** 2
        F[6:9, 0:3] = F_ap

        # F_av
        F_av = np.zeros((3, 3))
        F_av[0, 1] = 1 / rnh
        F_av[1, 0] = -1 / rmh
        F_av[2, 1] = -tan_lat / rnh
        F[6:9, 3:6] = F_av

        # F_ae, F_ag, F_as
        F[6:9, 6:9] = -CoordinateSystem.skew_symmetric(wie_n + wen_n)
        F[6:9, 9:12] = -pva.att.cbn
        F[6:9, 15:18] = -pva.att.cbn @ np.diag(omega)

        corr_time = self.options['imu_noise']['corr_time']
        if corr_time > 0:
            F[9:21, 9:21] -= np.eye(12) / corr_time

        return F

    def _compute_G_matrix(self, pva: PVA) -> np.ndarray:
        G = np.zeros((21, 18))
        G[3:6, 0:3] = pva.att.cbn
        G[6:9, 3:6] = pva.att.cbn
        G[9:21, 6:18] = np.eye(12)
        return G

    def get_nav_state(self) -> NavState:
        return NavState(pos=self.pvacur.pos, vel=self.pvacur.vel, euler=self.pvacur.att.euler, imuerror=self.imuerror)

    def _imu_compensate(self, imu: IMU) -> IMU:
        comp_imu = IMU(time=imu.time, dt=imu.dt)
        gyrscale_inv = 1.0 / (1.0 + self.imuerror.gyrscale)
        accscale_inv = 1.0 / (1.0 + self.imuerror.accscale)

        comp_imu.dtheta = imu.dtheta * gyrscale_inv - self.imuerror.gyrbias * imu.dt
        comp_imu.dvel = imu.dvel * accscale_inv - self.imuerror.accbias * imu.dt
        return comp_imu

    def _check_cov(self):
        diag_P = np.diag(self.P)
        if np.any(diag_P < 0):
            self.logger.warning(f"Negative covariance detected at time {self.imucur.time}. Diag: {diag_P}")

    def _load_options(self):
        imu_noise_cfg = config.get('imunoise')
        self.options = {
            'init_state': config.get('initial_state'),
            'init_state_std': config.get('initial_state_std'),
            'imu_noise': {
                'arw': np.radians(np.array(imu_noise_cfg['arw'])) / 60.0,
                'vrw': np.array(imu_noise_cfg['vrw']) / 60.0,
                'gbstd': np.radians(np.array(imu_noise_cfg['gbstd'])) / 3600.0,
                'abstd': np.array(imu_noise_cfg['abstd']) * 1e-5,
                'gsstd': np.array(imu_noise_cfg['gsstd']) * 1e-6,
                'asstd': np.array(imu_noise_cfg['asstd']) * 1e-6,
                'corr_time': imu_noise_cfg['corrtime'] * 3600.0
            },
            'ant_lever': config.get('ant_lever', [0.136, -0.301, -0.184])
        }

    def _initialize_covariance(self) -> np.ndarray:
        std = self.options['init_state_std']
        P = np.zeros((21, 21))
        P[0:3, 0:3] = np.diag(np.array(std['pos']) ** 2)
        P[3:6, 3:6] = np.diag(np.array(std['vel']) ** 2)
        P[6:9, 6:9] = np.diag(np.radians(np.array(std['att'])) ** 2)

        gb_std_val = std.get('gyrbias', [50.0, 50.0, 50.0])
        ab_std_val = std.get('accbias', [250.0, 250.0, 250.0])
        gs_std_val = std.get('gyrscale', [1000.0, 1000.0, 1000.0])
        as_std_val = std.get('accscale', [1000.0, 1000.0, 1000.0])

        P[9:12, 9:12] = np.diag((np.radians(np.array(gb_std_val)) / 3600.0) ** 2)
        P[12:15, 12:15] = np.diag((np.array(ab_std_val) * 1e-5) ** 2)
        P[15:18, 15:18] = np.diag((np.array(gs_std_val) * 1e-6) ** 2)
        P[18:21, 18:21] = np.diag((np.array(as_std_val) * 1e-6) ** 2)
        return P

    def _initialize_process_noise(self) -> np.ndarray:
        noise = self.options['imu_noise']
        Qc = np.zeros((18, 18))
        corr_time = noise['corr_time']

        Qc[0:3, 0:3] = np.diag(noise['vrw'] ** 2)
        Qc[3:6, 3:6] = np.diag(noise['arw'] ** 2)
        if corr_time > 0:
            Qc[6:9, 6:9] = np.diag(2 / corr_time * noise['gbstd'] ** 2)
            Qc[9:12, 9:12] = np.diag(2 / corr_time * noise['abstd'] ** 2)
            Qc[12:15, 12:15] = np.diag(2 / corr_time * noise['gsstd'] ** 2)
            Qc[15:18, 15:18] = np.diag(2 / corr_time * noise['asstd'] ** 2)
        return Qc