# multi_source_fusion/modules/eskf.py

import numpy as np
from dataclasses import dataclass, field
import logging

from core.coordinates import CoordinateSystem
from core.config import config
from core.types import PVA, Attitude, ImuError, IMU, GNSS, NavState


class _INSMech:
    """
    Python implementation of insmech.cpp - Final Corrected Version
    """

    @staticmethod
    def ins_mech(pvapre: PVA, imupre: IMU, imucur: IMU) -> PVA:
        dt = imucur.dt
        if dt <= 0: return pvapre

        pvacur = PVA(pos=pvapre.pos.copy(), vel=pvapre.vel.copy(), att=Attitude())
        pvacur.att.qbn = pvapre.att.qbn

        # 1. 姿态更新: 采用四元数更新法
        rm, rn, g = CoordinateSystem.get_radii_and_gravity(pvacur.pos)
        lat, h = pvacur.pos[0], pvacur.pos[2]

        wie_n = np.array([CoordinateSystem.WGS84_WIE * np.cos(lat), 0, -CoordinateSystem.WGS84_WIE * np.sin(lat)])
        wen_n = np.array([pvapre.vel[1] / (rn + h), -pvapre.vel[0] / (rm + h), -pvapre.vel[0] * np.tan(lat) / (rn + h)])

        alpha_nin = (wie_n + wen_n) * dt
        beta_bib = imucur.dtheta + np.cross(imupre.dtheta, imucur.dtheta) / 12.0

        q_nn = CoordinateSystem.rot_vec_to_quaternion(-alpha_nin)
        q_bb = CoordinateSystem.rot_vec_to_quaternion(beta_bib)

        q_bn_pre = pvapre.att.qbn
        q_bn_cur = CoordinateSystem.quaternion_multiply(q_nn, q_bn_pre)
        q_bn_cur = CoordinateSystem.quaternion_multiply(q_bn_cur, q_bb)

        pvacur.att.qbn = q_bn_cur
        pvacur.att.cbn = CoordinateSystem.quaternion_to_matrix(q_bn_cur)

        # 2. 速度更新
        mid_pos = 0.5 * (pvapre.pos + pvacur.pos)
        mid_vel_approx = 0.5 * (pvapre.vel + pvacur.vel)  # approximation
        rm_mid, rn_mid, g_mid = CoordinateSystem.get_radii_and_gravity(mid_pos)
        lat_mid, h_mid = mid_pos[0], mid_pos[2]

        wie_n_mid = np.array(
            [CoordinateSystem.WGS84_WIE * np.cos(lat_mid), 0, -CoordinateSystem.WGS84_WIE * np.sin(lat_mid)])
        wen_n_mid = np.array([mid_vel_approx[1] / (rn_mid + h_mid), -mid_vel_approx[0] / (rm_mid + h_mid),
                              -mid_vel_approx[1] * np.tan(lat_mid) / (rn_mid + h_mid)])

        d_vfb = imucur.dvel + 0.5 * np.cross(imucur.dtheta, imucur.dvel)
        cnn = np.eye(3) - CoordinateSystem.skew_symmetric(0.5 * (wie_n + wen_n) * dt)

        d_vfn = cnn @ pvapre.att.cbn @ d_vfb

        g_n = np.array([0, 0, g_mid])
        d_vgn = (g_n - np.cross(2 * wie_n_mid + wen_n_mid, mid_vel_approx)) * dt
        pvacur.vel = pvapre.vel + d_vfn + d_vgn

        # 3. 位置更新
        final_mid_vel = 0.5 * (pvapre.vel + pvacur.vel)
        dpos = np.zeros(3)
        dpos[0] = final_mid_vel[0] / (rm_mid + h_mid)
        dpos[1] = final_mid_vel[1] / ((rn_mid + h_mid) * np.cos(lat_mid))
        dpos[2] = final_mid_vel[2]
        pvacur.pos = pvapre.pos + np.array([dpos[0], dpos[1], -dpos[2]]) * dt

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

        self.imupre = IMU()
        self.imucur = IMU()
        self.gnssdata = None

    def add_imu_data(self, imu: IMU):
        self.imupre = self.imucur
        self.imucur = imu

    def add_gnss_data(self, gnss: GNSS):
        self.gnssdata = gnss

    def new_imu_process(self):
        compensated_imu = self._imu_compensate(self.imucur)
        self._ins_propagation(self.imupre, compensated_imu)
        self.pvapre = self.pvacur
        self.imupre = compensated_imu

    def _ins_propagation(self, imupre: IMU, imucur: IMU):
        self.pvacur = _INSMech.ins_mech(self.pvapre, imupre, imucur)

        # --- Stability Check ---
        if not np.all(np.isfinite(self.pvacur.pos)) or not np.all(np.isfinite(self.pvacur.vel)):
            self.logger.error(f"NaN or Inf detected in PVA state at time {imucur.time}. Halting propagation.")
            # In a real system, you might try to re-initialize or use other recovery methods.
            # Here we will just stop to prevent further corruption.
            return

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

        pos_ant = self.pvacur.pos + CoordinateSystem.DRi(self.pvacur.pos) @ self.pvacur.att.cbn @ ant_lever

        dz = CoordinateSystem.DR(self.pvacur.pos) @ (pos_ant - self.gnssdata.pos)

        H = np.zeros((3, 21))
        H[:, 0:3] = -np.eye(3)
        H[:, 6:9] = CoordinateSystem.skew_symmetric(self.pvacur.att.cbn @ ant_lever)

        R = np.diag(self.gnssdata.std ** 2)

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.dx = self.dx + K @ (dz - H @ self.dx)
        I = np.eye(21) - K @ H
        self.P = I @ self.P @ I.T + K @ R @ K.T
        self._check_cov()

        self._state_feedback()
        self.gnssdata = None

    def _state_feedback(self):
        # --- CRITICAL FIX: Correct feedback signs ---
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
        # --- THE FINAL, DEFINITIVE, CORRECTED F MATRIX ---
        F = np.zeros((21, 21))

        lat, h = pva.pos[0], pva.pos[2]
        v_n, v_e, v_d = pva.vel[0], pva.vel[1], pva.vel[2]

        accel = imu.dvel / imu.dt if imu.dt > 0 else np.zeros(3)
        omega = imu.dtheta / imu.dt if imu.dt > 0 else np.zeros(3)

        rm, rn, g = CoordinateSystem.get_radii_and_gravity(pva.pos)
        rmh, rnh = rm + h, rn + h

        wie_n = np.array([CoordinateSystem.WGS84_WIE * np.cos(lat), 0, -CoordinateSystem.WGS84_WIE * np.sin(lat)])
        wen_n = np.array([v_e / rnh, -v_n / rmh, -v_e * np.tan(lat) / rnh])

        # F_pp (Rows 0-2, Cols 0-2)
        Fpp = np.zeros((3, 3))
        Fpp[0, 0] = -v_d / rmh
        Fpp[0, 2] = v_n / rmh ** 2
        Fpp[1, 0] = v_e * np.tan(lat) / rnh
        Fpp[1, 2] = -v_e / rnh ** 2
        F[0:3, 0:3] = Fpp

        # F_pv (Rows 0-2, Cols 3-5)
        Fpv = np.zeros((3, 3))
        Fpv[0, 0] = 1 / rmh
        Fpv[1, 1] = 1 / (rnh * np.cos(lat))
        Fpv[2, 2] = -1
        F[0:3, 3:6] = Fpv

        # F_vp (Rows 3-5, Cols 0-2)
        Fvp = np.zeros((3, 3))
        Fvp[0, 0] = -2 * v_e * CoordinateSystem.WGS84_WIE * np.cos(lat) - v_e ** 2 / (rnh * np.cos(lat) ** 2)
        Fvp[0, 2] = v_n * v_d / rmh ** 2 - v_e ** 2 * np.tan(lat) / rnh ** 2
        Fvp[1, 0] = 2 * CoordinateSystem.WGS84_WIE * (v_n * np.cos(lat) - v_d * np.sin(lat)) + v_n * v_e / (
                    rnh * np.cos(lat) ** 2)
        Fvp[1, 2] = (v_n * v_e * np.tan(lat) + v_e * v_d) / rnh ** 2
        Fvp[2, 0] = 2 * CoordinateSystem.WGS84_WIE * v_e * np.sin(lat)
        Fvp[2, 2] = -(v_e ** 2 / rnh ** 2 + v_n ** 2 / rmh ** 2) + 2 * g / (np.sqrt(rm * rn) + h)
        F[3:6, 0:3] = Fvp

        # F_vv (Rows 3-5, Cols 3-5)
        F[3:6, 3:6] = -CoordinateSystem.skew_symmetric(2 * wie_n + wen_n)

        # F_ve (Rows 3-5, Cols 6-8)
        F[3:6, 6:9] = CoordinateSystem.skew_symmetric(pva.att.cbn @ accel)

        # F_ap (Rows 6-8, Cols 0-2)
        Fap = np.zeros((3, 3))
        Fap[0, 0] = -CoordinateSystem.WGS84_WIE * np.sin(lat)
        Fap[0, 2] = v_n / rmh ** 2
        Fap[1, 2] = -v_e / rnh ** 2
        Fap[2, 0] = -CoordinateSystem.WGS84_WIE * np.cos(lat) - v_e / (rnh * np.cos(lat) ** 2)
        Fap[2, 2] = -v_e * np.tan(lat) / rnh ** 2
        F[6:9, 0:3] = Fap

        # F_av (Rows 6-8, Cols 3-5)
        Fav = np.zeros((3, 3))
        Fav[0, 0] = 1 / rmh
        Fav[1, 0] = -1 / rnh
        Fav[2, 1] = -np.tan(lat) / rnh
        F[6:9, 3:6] = Fav

        # F_ae (Rows 6-8, Cols 6-8)
        F[6:9, 6:9] = -CoordinateSystem.skew_symmetric(wie_n + wen_n)

        # IMU error couplings
        F[3:6, 12:15] = pva.att.cbn
        F[3:6, 18:21] = pva.att.cbn @ np.diag(accel)
        F[6:9, 9:12] = -pva.att.cbn
        F[6:9, 15:18] = -pva.att.cbn @ np.diag(omega)

        corr_time = self.options['imu_noise']['corr_time']
        if corr_time > 0: F[9:21, 9:21] -= np.eye(12) / corr_time

        return F

    def _compute_G_matrix(self, pva: PVA) -> np.ndarray:
        # ... (unchanged) ...
        G = np.zeros((21, 18))
        G[3:6, 0:3] = pva.att.cbn
        G[6:9, 3:6] = pva.att.cbn
        G[9:21, 6:18] = np.eye(12)
        return G

    def get_nav_state(self) -> NavState:
        # ... (unchanged) ...
        return NavState(pos=self.pvacur.pos, vel=self.pvacur.vel, euler=self.pvacur.att.euler, imuerror=self.imuerror)

    def _imu_compensate(self, imu: IMU) -> IMU:
        # ... (unchanged) ...
        comp_imu = IMU(time=imu.time, dt=imu.dt)
        comp_imu.dtheta = (imu.dtheta - self.imuerror.gyrbias * imu.dt) / (1.0 + self.imuerror.gyrscale)
        comp_imu.dvel = (imu.dvel - self.imuerror.accbias * imu.dt) / (1.0 + self.imuerror.accscale)
        return comp_imu

    def _check_cov(self):
        # ... (unchanged) ...
        if np.any(np.diag(self.P) < 0):
            self.logger.warning(f"检测到负协方差，时间: {self.imucur.time}")

    def _load_options(self):
        # ... (unchanged) ...
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
            'ant_lever': config.get('antlever', [0, 0, 0])
        }

    def _initialize_covariance(self) -> np.ndarray:
        # ... (unchanged) ...
        std = self.options['init_state_std']
        P = np.zeros((21, 21))
        P[0:3, 0:3] = np.diag(np.array(std['pos']) ** 2)
        P[3:6, 3:6] = np.diag(np.array(std['vel']) ** 2)
        P[6:9, 6:9] = np.diag(np.radians(np.array(std['att'])) ** 2)
        gb_std_val = std.get('gyrbias', self.options['imu_noise']['gbstd'] * 3600 / np.pi)
        ab_std_val = std.get('accbias', self.options['imu_noise']['abstd'] / 1e-5)
        gs_std_val = std.get('gyrscale', self.options['imu_noise']['gsstd'] / 1e-6)
        as_std_val = std.get('accscale', self.options['imu_noise']['asstd'] / 1e-6)
        P[9:12, 9:12] = np.diag((np.radians(np.array(gb_std_val)) / 3600.0) ** 2)
        P[12:15, 12:15] = np.diag((np.array(ab_std_val) * 1e-5) ** 2)
        P[15:18, 15:18] = np.diag((np.array(gs_std_val) * 1e-6) ** 2)
        P[18:21, 18:21] = np.diag((np.array(as_std_val) * 1e-6) ** 2)
        return P

    def _initialize_process_noise(self) -> np.ndarray:
        # ... (unchanged) ...
        noise = self.options['imu_noise']
        Qc = np.zeros((18, 18))
        Qc[0:3, 0:3] = np.diag(noise['vrw'] ** 2)
        Qc[3:6, 3:6] = np.diag(noise['arw'] ** 2)
        Qc[6:9, 6:9] = np.diag(2 / noise['corr_time'] * noise['gbstd'] ** 2)
        Qc[9:12, 9:12] = np.diag(2 / noise['corr_time'] * noise['abstd'] ** 2)
        Qc[12:15, 12:15] = np.diag(2 / noise['corr_time'] * noise['gsstd'] ** 2)
        Qc[15:18, 15:18] = np.diag(2 / noise['corr_time'] * noise['asstd'] ** 2)
        return Qc