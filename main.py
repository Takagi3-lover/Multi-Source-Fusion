# multi_source_fusion/main.py

import os
import sys
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from core.config import config
from data.loaders import load_imu_data, load_gnss_data
from modules.eskf import ErrorStateKalmanFilter, IMU, GNSS
from core.types import NavState

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=== 多源融合定位系统启动 (KF-GINS移植版) ===")

    imu_df = load_imu_data(config.get('data_paths.imu_path'))
    gnss_df = load_gnss_data(config.get('data_paths.gnss_path'))

    if imu_df.empty:
        logger.error("IMU数据为空，程序终止。")
        return

    eskf = ErrorStateKalmanFilter()

    logger.info("=== 开始轨迹处理 ===")
    results = []

    # 初始化第一个IMU数据
    first_imu_row = imu_df.iloc[0]
    eskf.add_imu_data(IMU(time=first_imu_row.timestamp))

    gnss_queue = [GNSS(time=row.timestamp, pos=np.array([row.lat, row.lon, row.h]),
                       std=np.array([row.std_lat, row.std_lon, row.std_h])) for row in gnss_df.itertuples(index=False)]

    # --- CRITICAL FIX: Correctly iterate from the second row ---
    imu_iterator = imu_df.iloc[1:].itertuples(index=False)

    for imu_row in tqdm(imu_iterator, total=len(imu_df) - 1, desc="Processing Data"):

        current_imu = IMU(time=imu_row.timestamp, dt=imu_row.timestamp - eskf.imucur.time,
                          dtheta=np.array([imu_row.d_angle_x, imu_row.d_angle_y, imu_row.d_angle_z]),
                          dvel=np.array([imu_row.d_vel_x, imu_row.d_vel_y, imu_row.d_vel_z]))

        time_align_err = 1.0 / config.get('sensor_params.imu.rate_hz', 200.0) * 0.2

        # 检查是否有GNSS数据需要在当前IMU时间之前处理
        if gnss_queue and gnss_queue[0].time < current_imu.time + time_align_err:
            gnss_data = gnss_queue.pop(0)

            # 内插IMU到GNSS时刻
            imu_prev = eskf.imucur
            if current_imu.dt > 0:
                lamda = (gnss_data.time - imu_prev.time) / current_imu.dt

                mid_imu = IMU(time=gnss_data.time, dt=gnss_data.time - imu_prev.time)
                mid_imu.dtheta = current_imu.dtheta * lamda
                mid_imu.dvel = current_imu.dvel * lamda

                eskf.add_imu_data(mid_imu)
                eskf.new_imu_process()

                eskf.add_gnss_data(gnss_data)
                eskf.gnss_update()

                remaining_dt = current_imu.time - mid_imu.time
                if remaining_dt > 1e-6:
                    remaining_imu = IMU(time=current_imu.time, dt=remaining_dt,
                                        dtheta=current_imu.dtheta * (1 - lamda),
                                        dvel=current_imu.dvel * (1 - lamda))
                    eskf.add_imu_data(remaining_imu)
                    eskf.new_imu_process()
        else:
            eskf.add_imu_data(current_imu)
            eskf.new_imu_process()

        # 保存结果 (NED to ENU for output)
        state = eskf.get_nav_state()
        results.append({
            'timestamp': eskf.imucur.time,
            'lat_deg': np.degrees(state.pos[0]), 'lon_deg': np.degrees(state.pos[1]), 'h': state.pos[2],
            'v_n': state.vel[0], 'v_e': state.vel[1], 'v_d': state.vel[2],
            'roll_deg': np.degrees(state.euler[0]), 'pitch_deg': np.degrees(state.euler[1]),
            'yaw_deg': np.degrees(state.euler[2]),
        })

    logger.info("=== 处理完成，正在保存结果 ===")
    if results:
        output_dir = config.get('data_paths.output_path', 'results/')
        os.makedirs(output_dir, exist_ok=True)
        results_df = pd.DataFrame(results)
        output_file = os.path.join(output_dir, 'final_trajectory_kfgins_py.csv')
        results_df.to_csv(output_file, index=False)
        logger.info(f"轨迹已保存至: {output_file}")
    else:
        logger.warning("没有生成任何有效结果。")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.critical(f"程序执行过程中发生未处理的严重错误: {e}", exc_info=True)