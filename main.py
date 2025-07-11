# multi_source_fusion/main.py

import os
import numpy as np
import pandas as pd
import torch
from typing import List

from core.config import config
from core.types import SystemState
from data.loaders import load_map_data
from data.dataset import create_synchronized_dataset
from modules.ekf import MTINN_EKF
from modules.map_matcher import MapMatcher
from inference.predictor import MTINNPredictor


def main():
    """
    主执行函数，运行整个多源融合定位流程。
    """
    print("=== 多源融合定位系统启动 ===")

    # 1. 初始化
    print("--- 系统初始化 ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载数据路径
    imu_path = config.get('data_paths.imu_path')
    gnss_path = config.get('data_paths.gnss_path')
    odo_path = config.get('data_paths.odo_path')
    map_path = config.get('data_paths.map_path')

    print(f"数据路径:")
    print(f"  IMU: {imu_path}")
    print(f"  GNSS: {gnss_path}")
    print(f"  ODO: {odo_path}")
    print(f"  地图: {map_path}")

    # 检查数据文件是否存在
    missing_files = []
    for name, path in [("IMU", imu_path), ("GNSS", gnss_path), ("ODO", odo_path), ("地图", map_path)]:
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")

    if missing_files:
        print("错误: 以下数据文件不存在:")
        for file in missing_files:
            print(f"  - {file}")
        print("请检查配置文件中的路径设置。")
        return

    # 加载数据
    try:
        print("正在加载数据...")
        nav_frames = create_synchronized_dataset(imu_path, gnss_path, odo_path)
        map_df = load_map_data(map_path)

        if len(nav_frames) == 0:
            print("错误: 导航数据为空")
            return

        print(f"成功加载 {len(nav_frames)} 个导航帧")
        print(f"成功加载 {len(map_df)} 个地图点")

    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    # 初始化预测器
    model_path = os.path.join(config.get('data_paths.output_path', 'results/'), 'models/best_mtinn_model.pth')
    if not os.path.exists(model_path):
        print(f"错误: 找不到训练好的模型 '{model_path}'。")
        print("请先运行训练脚本: python -m training.train_mtinn")
        return

    try:
        predictor = MTINNPredictor(model_path, device)
        print("MTINN预测器初始化成功")
    except Exception as e:
        print(f"预测器初始化失败: {e}")
        return

    # 初始化地图匹配器
    map_matcher = None
    if config.get('map_matching.enable', True) and not map_df.empty:
        try:
            map_matcher = MapMatcher(map_df)
            print("地图匹配器初始化成功")
        except Exception as e:
            print(f"地图匹配器初始化失败: {e}")
            print("将在没有地图匹配的情况下运行")

    # 定义初始状态
    try:
        init_pos_deg = config.get('initial_state.init_pos', [30.5, 114.3, 30.0])
        init_att_deg = config.get('initial_state.init_att', [0.0, 0.0, 90.0])
        init_vel = config.get('initial_state.init_vel', [0.0, 0.0, 0.0])

        init_att_std_deg = config.get('initial_state.init_att_std', [1.0, 1.0, 5.0])
        init_vel_std = config.get('initial_state.init_vel_std', [0.1, 0.1, 0.1])
        init_pos_std = config.get('initial_state.init_pos_std', [10.0, 10.0, 20.0])

        # 构建初始协方差矩阵
        init_cov_diag = (
                [np.radians(std) ** 2 for std in init_att_std_deg] +  # 姿态方差
                [std ** 2 for std in init_vel_std] +  # 速度方差
                [std ** 2 for std in init_pos_std]  # 位置方差
        )

        initial_state = SystemState(
            timestamp=nav_frames[0].timestamp,
            attitude=np.radians(init_att_deg),
            velocity=np.array(init_vel),
            position=np.array([np.radians(init_pos_deg[0]), np.radians(init_pos_deg[1]), init_pos_deg[2]]),
            covariance=np.diag(init_cov_diag)
        )

        print("初始状态设置成功")

    except Exception as e:
        print(f"初始状态设置失败: {e}")
        return

    # 初始化EKF
    try:
        ekf = MTINN_EKF(predictor, initial_state)
        print("EKF滤波器初始化成功")
    except Exception as e:
        print(f"EKF初始化失败: {e}")
        return

    # 2. 主处理循环
    print("--- 开始处理轨迹 ---")
    results = []
    feedback_error = np.zeros(3)

    # GNSS更新频率控制
    imu_rate = config.get('sensor_params.imu.rate_hz', 100.0)
    gnss_rate = config.get('sensor_params.gnss.rate_hz', 1.0)
    gnss_update_interval = max(1, int(imu_rate / gnss_rate))

    for i in range(1, len(nav_frames)):
        try:
            prev_frame = nav_frames[i - 1]
            current_frame = nav_frames[i]

            dt = current_frame.timestamp - prev_frame.timestamp

            # 检查时间间隔的合理性
            if dt <= 0 or dt > 1.0:  # 时间间隔应该在合理范围内
                print(f"警告: 时间间隔异常 dt={dt:.6f}s，跳过此帧")
                continue

            # a. EKF 预测步骤
            ekf.predict(current_frame.imu_data, current_frame.odo_data, dt, feedback_error)

            # b. EKF 更新步骤 (如果GNSS可用)
            if i % gnss_update_interval == 0 and current_frame.gnss_data is not None:
                ekf.update(current_frame.gnss_data)

            # c. 地图匹配与反馈
            ekf_state = ekf.get_state()
            final_state = ekf_state

            if map_matcher is not None:
                try:
                    matched_state = map_matcher.match(ekf_state)
                    final_state = matched_state

                    # 计算反馈误差
                    if config.get('feedback_loop.enable', True):
                        pos_error = matched_state.position - ekf_state.position
                        damping = config.get('feedback_loop.damping_factor', 0.5)
                        feedback_error = pos_error * damping

                except Exception as e:
                    print(f"地图匹配失败: {e}")
                    final_state = ekf_state

            # d. 存储结果
            result_dict = {
                'timestamp': final_state.timestamp,
                'lat_deg': np.degrees(final_state.position[0]),
                'lon_deg': np.degrees(final_state.position[1]),
                'h': final_state.position[2],
                'v_e': final_state.velocity[0],
                'v_n': final_state.velocity[1],
                'v_u': final_state.velocity[2],
                'roll_deg': np.degrees(final_state.attitude[0]),
                'pitch_deg': np.degrees(final_state.attitude[1]),
                'yaw_deg': np.degrees(final_state.attitude[2]),
            }
            results.append(result_dict)

            # 进度显示
            if i % 1000 == 0:
                print(f"已处理 {i}/{len(nav_frames)} 帧 ({100 * i / len(nav_frames):.1f}%)")

        except Exception as e:
            print(f"处理第 {i} 帧时出错: {e}")
            continue

    # 3. 保存输出
    print("--- 处理完成，正在保存结果 ---")

    if not results:
        print("错误: 没有成功处理的结果")
        return

    try:
        output_path = config.get('data_paths.output_path', 'results/')
        os.makedirs(output_path, exist_ok=True)

        result_df = pd.DataFrame(results)
        output_file = os.path.join(output_path, 'final_trajectory.csv')
        result_df.to_csv(output_file, index=False)

        print(f"轨迹已保存至: {output_file}")
        print(f"共保存 {len(results)} 个轨迹点")

        # 显示统计信息
        print("\n--- 处理统计 ---")
        print(f"轨迹时长: {results[-1]['timestamp'] - results[0]['timestamp']:.1f} 秒")
        print(f"平均速度: {np.sqrt(np.mean([r['v_e'] ** 2 + r['v_n'] ** 2 for r in results])):.2f} m/s")
        print(f"高度范围: {min(r['h'] for r in results):.1f} ~ {max(r['h'] for r in results):.1f} m")

    except Exception as e:
        print(f"保存结果失败: {e}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断程序执行")
    except Exception as e:
        print(f"程序执行出现未预期错误: {e}")
        import traceback

        traceback.print_exc()