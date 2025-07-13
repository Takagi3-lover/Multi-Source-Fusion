# multi_source_fusion/main.py

import os
import sys
import numpy as np
import pandas as pd
import torch
from typing import List
import logging

# --- 关键设置 ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)
# -----------------

from core.config import config
from core.types import SystemState, NavFrame
from data import loaders
from modules.eskf import ErrorStateKalmanFilter, FusionMode
from modules.map_matcher import MapMatcher
from inference.predictor import MTINNPredictor

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_inference_data(imu_path: str, gnss_path: str, odo_path: str) -> List[NavFrame]:
    """
    为推理过程准备和同步数据 - 采用KF-GINS标准做法，不进行插值
    """
    logger.info("=== 准备推理数据 ===")
    imu_df = loaders.load_imu_data(imu_path)
    gnss_df = loaders.load_gnss_data(gnss_path)

    # 检查是否启用里程计
    odo_enabled = config.get('sensor_params.odo.enable', False)
    if odo_enabled:
        odo_df = loaders.load_odo_data(odo_path)
        if not odo_df.empty:
            logger.info(f"里程计已启用，加载了 {len(odo_df)} 个数据点")
        else:
            logger.warning("里程计启用但数据文件为空，将禁用里程计")
            odo_enabled = False
            odo_df = pd.DataFrame()
    else:
        odo_df = pd.DataFrame()
        logger.info("里程计已禁用")

    if imu_df.empty:
        raise ValueError("IMU数据为空，无法继续。")

    # === 标准做法：以IMU为主时间轴，不进行插值 ===
    logger.info("正在构建导航帧序列...")

    # 为GNSS和ODO数据建立时间索引
    gnss_dict = {}
    if not gnss_df.empty:
        for _, row in gnss_df.iterrows():
            gnss_dict[row['timestamp']] = {
                'lat': row['lat'], 'lon': row['lon'], 'h': row['h'],
                'std_lat': row.get('std_lat', 1.0),
                'std_lon': row.get('std_lon', 1.0),
                'std_h': row.get('std_h', 2.0)
            }
        logger.info(f"GNSS数据已索引: {len(gnss_dict)} 个观测点")

    odo_dict = {}
    if odo_enabled and not odo_df.empty:
        for _, row in odo_df.iterrows():
            odo_dict[row['timestamp']] = {'velocity': row['velocity']}
        logger.info(f"里程计数据已索引: {len(odo_dict)} 个观测点")

    # 检查关键传感器数据
    required_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
    missing_cols = [col for col in required_cols if col not in imu_df.columns]
    if missing_cols:
        raise ValueError(f"缺少必要的IMU数据列: {missing_cols}")

    # 数据质量检查
    logger.info("进行数据质量检查...")

    # 检查IMU数据范围
    for col in ['gx', 'gy', 'gz']:
        data_range = imu_df[col].max() - imu_df[col].min()
        if data_range > 10:  # 陀螺仪输出超过10 rad/s可能异常
            logger.warning(f"{col} 数据范围异常: {data_range:.2f} rad/s")

    for col in ['ax', 'ay', 'az']:
        data_range = imu_df[col].max() - imu_df[col].min()
        if data_range > 50:  # 加速度计输出超过50 m/s²可能异常
            logger.warning(f"{col} 数据范围异常: {data_range:.2f} m/s²")

    # === 构建NavFrame序列 - 以IMU时间戳为主轴 ===
    nav_frames: List[NavFrame] = []
    gnss_count = 0
    odo_count = 0

    # 定义时间匹配阈值
    gnss_time_threshold = 0.5 / config.get('sensor_params.gnss.rate_hz', 1.0)  # GNSS时间阈值
    odo_time_threshold = 0.5 / config.get('sensor_params.odo.rate_hz', 200.0)  # ODO时间阈值

    for _, imu_row in imu_df.iterrows():
        timestamp = imu_row['timestamp']

        # IMU数据（每帧都有）
        imu_data = {
            'ax': imu_row['ax'], 'ay': imu_row['ay'], 'az': imu_row['az'],
            'gx': imu_row['gx'], 'gy': imu_row['gy'], 'gz': imu_row['gz']
        }

        # 查找最近的GNSS数据
        gnss_data = None
        if gnss_dict:
            closest_gnss_time = min(gnss_dict.keys(), key=lambda x: abs(x - timestamp))
            if abs(closest_gnss_time - timestamp) <= gnss_time_threshold:
                gnss_data = gnss_dict[closest_gnss_time]
                gnss_count += 1

        # 查找最近的ODO数据
        odo_data = None
        if odo_enabled and odo_dict:
            closest_odo_time = min(odo_dict.keys(), key=lambda x: abs(x - timestamp))
            if abs(closest_odo_time - timestamp) <= odo_time_threshold:
                odo_data = odo_dict[closest_odo_time]
                odo_count += 1

        frame = NavFrame(
            timestamp=timestamp,
            imu_data=imu_data,
            odo_data=odo_data,
            gnss_data=gnss_data
        )
        nav_frames.append(frame)

    logger.info(f"推理数据准备完成:")
    logger.info(f"  - 总帧数: {len(nav_frames)} (以IMU频率为主)")
    logger.info(f"  - GNSS观测: {gnss_count} 帧 ({100 * gnss_count / len(nav_frames):.1f}%)")
    logger.info(f"  - 里程计状态: {'启用' if odo_enabled else '禁用'}")
    if odo_enabled:
        logger.info(f"  - 里程计观测: {odo_count} 帧 ({100 * odo_count / len(nav_frames):.1f}%)")
    logger.info(f"  - 数据时长: {nav_frames[-1].timestamp - nav_frames[0].timestamp:.1f} 秒")
    logger.info(f"  - 平均IMU频率: {len(nav_frames) / (nav_frames[-1].timestamp - nav_frames[0].timestamp):.1f} Hz")

    return nav_frames


def initialize_predictor(fusion_mode: FusionMode, device: torch.device):
    """
    根据融合模式初始化MTINN预测器
    """
    needs_model = fusion_mode in [FusionMode.MODEL_ONLY, FusionMode.WEIGHTED_FUSION, FusionMode.ADAPTIVE_FUSION]

    if not needs_model:
        logger.info("当前融合模式不需要神经网络模型")
        return None

    model_path = os.path.join(config.get('data_paths.model_save_dir', 'models/'), 'best_mtinn_model.pth')

    if not os.path.exists(model_path):
        logger.warning(f"模型文件不存在: {model_path}")
        if fusion_mode == FusionMode.MODEL_ONLY:
            raise FileNotFoundError(f"MODEL_ONLY模式下必须提供有效的模型文件: {model_path}")
        else:
            logger.info("将回退到纯数学模型")
            return None

    try:
        predictor = MTINNPredictor(model_path, device)
        logger.info(f"MTINN预测器已成功加载: {model_path}")
        return predictor
    except Exception as e:
        logger.error(f"加载MTINN预测器失败: {e}")
        if fusion_mode == FusionMode.MODEL_ONLY:
            raise RuntimeError(f"MODEL_ONLY模式下模型加载失败: {e}")
        else:
            logger.info("将回退到纯数学模型")
            return None


def initialize_system_state(nav_frames: List[NavFrame]) -> SystemState:
    """
    初始化系统状态
    """
    logger.info("=== 初始化系统状态 ===")

    # 寻找第一个有效的GNSS点作为初始位置
    first_gnss_frame = next((frame for frame in nav_frames if frame.gnss_data is not None), None)

    if first_gnss_frame:
        init_pos = [
            first_gnss_frame.gnss_data['lat'],
            first_gnss_frame.gnss_data['lon'],
            first_gnss_frame.gnss_data['h']
        ]
        init_pos_deg_str = f"[{np.degrees(init_pos[0]):.6f}°, {np.degrees(init_pos[1]):.6f}°, {init_pos[2]:.2f}m]"
        logger.info(f"使用第一个有效GNSS点作为初始位置: {init_pos_deg_str}")
    else:
        init_pos_deg = config.get('initial_state.init_pos')
        init_pos = [np.radians(init_pos_deg[0]), np.radians(init_pos_deg[1]), init_pos_deg[2]]
        init_pos_deg_str = f"[{init_pos_deg[0]:.6f}°, {init_pos_deg[1]:.6f}°, {init_pos_deg[2]:.2f}m]"
        logger.warning(f"未找到GNSS点，使用配置文件中的初始位置: {init_pos_deg_str}")

    # 从配置获取初始状态参数
    init_att_deg = config.get('initial_state.init_att')
    init_vel = config.get('initial_state.init_vel')
    init_att_std_deg = config.get('initial_state.init_att_std')
    init_vel_std = config.get('initial_state.init_vel_std')
    init_pos_std = config.get('initial_state.init_pos_std')

    # 构建初始协方差矩阵（这里只是为了兼容性，ESKF内部会重新构建）
    init_cov_diag = (
            [np.radians(std) ** 2 for std in init_att_std_deg] +
            [std ** 2 for std in init_vel_std] +
            [std ** 2 for std in init_pos_std]
    )

    initial_state = SystemState(
        timestamp=nav_frames[0].timestamp,
        attitude=np.radians(init_att_deg),
        velocity=np.array(init_vel),
        position=np.array(init_pos),
        covariance=np.diag(init_cov_diag)
    )

    logger.info(f"初始状态设置完成:")
    logger.info(f"  - 初始姿态: roll={init_att_deg[0]:.1f}°, pitch={init_att_deg[1]:.1f}°, yaw={init_att_deg[2]:.1f}°")
    logger.info(f"  - 初始速度: vE={init_vel[0]:.1f}, vN={init_vel[1]:.1f}, vU={init_vel[2]:.1f} m/s")
    logger.info(f"  - 初始位置: {init_pos_deg_str}")

    return initial_state


def print_fusion_strategy_info():
    """
    打印融合策略信息
    """
    fusion_mode, model_weight = config.validate_fusion_strategy()
    fusion_mode_enum = FusionMode(fusion_mode)

    logger.info("=== 融合策略配置 ===")
    logger.info(f"融合模式: {fusion_mode}")

    if fusion_mode_enum == FusionMode.MATH_ONLY:
        logger.info("  - 完全依赖Error State Kalman Filter (ESKF)")
        logger.info("  - 不使用神经网络模型")
    elif fusion_mode_enum == FusionMode.MODEL_ONLY:
        logger.info("  - 完全依赖MTINN神经网络模型")
        logger.info("  - 不使用传统ESKF")
    elif fusion_mode_enum == FusionMode.WEIGHTED_FUSION:
        logger.info(f"  - 加权融合: {(1 - model_weight) * 100:.1f}% ESKF + {model_weight * 100:.1f}% 神经网络模型")
    elif fusion_mode_enum == FusionMode.ADAPTIVE_FUSION:
        adaptive_params = config.get('fusion_strategy.adaptive_params', {})
        min_weight = adaptive_params.get('min_model_weight', 0.0)
        max_weight = adaptive_params.get('max_model_weight', 0.3)
        logger.info(f"  - 自适应融合: 模型权重范围 [{min_weight:.1f}, {max_weight:.1f}]")
        logger.info(f"  - 根据系统不确定性动态调整权重")

    return fusion_mode_enum


def main():
    """主执行函数"""
    logger.info("=== 多源融合定位系统启动 (KF-GINS标准版本) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"计算设备: {device}")

    # --- 1. 打印融合策略信息 ---
    fusion_mode = print_fusion_strategy_info()

    # --- 2. 数据加载 ---
    logger.info("=== 数据加载阶段 ===")
    imu_path = config.get('data_paths.imu_path')
    gnss_path = config.get('data_paths.gnss_path')
    odo_path = config.get('data_paths.odo_path')
    map_path = config.get('data_paths.map_path')

    logger.info(f"数据文件路径:")
    logger.info(f"  - IMU: {imu_path}")
    logger.info(f"  - GNSS: {gnss_path}")
    logger.info(f"  - 里程计: {odo_path} ({'启用' if config.get('sensor_params.odo.enable', False) else '禁用'})")
    logger.info(f"  - 地图: {map_path}")

    try:
        nav_frames = prepare_inference_data(imu_path, gnss_path, odo_path)
        map_df = loaders.load_map_data(map_path)
    except Exception as e:
        logger.error(f"数据加载失败: {e}")
        return

    # --- 3. 组件初始化 ---
    logger.info("=== 组件初始化阶段 ===")

    # 初始化预测器
    try:
        predictor = initialize_predictor(fusion_mode, device)
    except Exception as e:
        logger.error(f"预测器初始化失败: {e}")
        return

    # 初始化地图匹配器（暂时禁用）
    map_matcher = None
    if not map_df.empty and config.get('map_matching.enable', False):
        try:
            map_matcher = MapMatcher(map_df)
            logger.info(f"地图匹配器已初始化，地图点数: {len(map_df)}")
        except Exception as e:
            logger.warning(f"地图匹配器初始化失败: {e}")

    # 初始化系统状态
    try:
        initial_state = initialize_system_state(nav_frames)
    except Exception as e:
        logger.error(f"系统状态初始化失败: {e}")
        return

    # 初始化ESKF
    try:
        eskf = ErrorStateKalmanFilter(predictor, initial_state)
        logger.info("Error State Kalman Filter (KF-GINS风格) 初始化完成")
    except Exception as e:
        logger.error(f"ESKF初始化失败: {e}")
        return

    # --- 4. 主处理循环 ---
    logger.info("=== 开始轨迹处理 ===")
    results = []
    feedback_error = np.zeros(3)

    # 统计变量
    gnss_updates = 0
    odo_updates = 0
    processing_errors = 0

    logger.info(f"开始处理 {len(nav_frames)} 帧数据...")

    for i in range(1, len(nav_frames)):
        current_frame = nav_frames[i]
        prev_frame = nav_frames[i - 1]
        dt = current_frame.timestamp - prev_frame.timestamp

        # 检查时间间隔的合理性
        if dt <= 0 or dt > 1.0:
            logger.warning(f"异常的时间间隔 dt={dt:.6f}s, 跳过第 {i} 帧")
            processing_errors += 1
            continue

        # === 1. ESKF预测步骤（每帧都执行） ===
        try:
            eskf.predict(current_frame.imu_data, current_frame.odo_data, dt, feedback_error)
        except Exception as e:
            logger.error(f"第 {i} 帧ESKF预测步骤出错: {e}")
            processing_errors += 1
            continue

        # === 2. GNSS更新步骤（仅当有GNSS数据时） ===
        if current_frame.gnss_data is not None:
            try:
                eskf.update(current_frame.gnss_data)
                gnss_updates += 1
                logger.debug(f"第 {i} 帧: GNSS更新执行")
            except Exception as e:
                logger.error(f"第 {i} 帧GNSS更新步骤出错: {e}")
                processing_errors += 1

        # === 3. 里程计约束（仅当有ODO数据时） ===
        if current_frame.odo_data is not None:
            odo_updates += 1
            logger.debug(f"第 {i} 帧: 里程计约束已应用")

        # 获取当前状态
        eskf_state = eskf.get_state()
        final_state = eskf_state

        # 地图匹配（如果启用）
        if map_matcher is not None:
            try:
                matched_state = map_matcher.match(eskf_state)
                if matched_state is not None:
                    final_state = matched_state

                    # 反馈回路（如果启用）
                    if config.get('feedback_loop.enable', False):
                        pos_error = matched_state.position - eskf_state.position
                        damping = config.get('feedback_loop.damping_factor', 0.5)
                        feedback_error = pos_error * damping
                    else:
                        feedback_error = np.zeros(3)
                else:
                    feedback_error = np.zeros(3)
            except Exception as e:
                logger.error(f"第 {i} 帧地图匹配出错: {e}")
                processing_errors += 1
                feedback_error = np.zeros(3)

        # 保存结果
        results.append({
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
        })

        # 定期打印进度和状态
        if i % 2000 == 0 or i == len(nav_frames) - 1:
            progress = 100 * i / len(nav_frames)
            logger.info(f"处理进度: {i}/{len(nav_frames)} ({progress:.1f}%)")
            logger.info(f"  当前位置: lat={np.degrees(final_state.position[0]):.6f}°, "
                        f"lon={np.degrees(final_state.position[1]):.6f}°, h={final_state.position[2]:.2f}m")
            logger.info(f"  当前姿态: roll={np.degrees(final_state.attitude[0]):.2f}°, "
                        f"pitch={np.degrees(final_state.attitude[1]):.2f}°, yaw={np.degrees(final_state.attitude[2]):.2f}°")
            logger.info(
                f"  当前速度: vE={final_state.velocity[0]:.2f}, vN={final_state.velocity[1]:.2f}, vU={final_state.velocity[2]:.2f} m/s")
            logger.info(f"  GNSS更新: {gnss_updates}, 里程计更新: {odo_updates}")

    # --- 5. 保存结果和统计 ---
    logger.info("=== 处理完成，正在保存结果 ===")

    if not results:
        logger.error("没有生成任何有效结果")
        return

    output_path = config.get('data_paths.output_path', 'results/')
    os.makedirs(output_path, exist_ok=True)

    results_df = pd.DataFrame(results)
    output_file = os.path.join(output_path, 'final_trajectory_eskf_kfgins.csv')
    results_df.to_csv(output_file, index=False)

    logger.info(f"轨迹已保存至: {output_file}")

    # 处理统计
    logger.info(f"=== 处理统计 ===")
    logger.info(f"总处理帧数: {len(nav_frames)}")
    logger.info(f"有效结果帧数: {len(results)}")
    logger.info(f"GNSS更新次数: {gnss_updates}")
    logger.info(f"里程计更新次数: {odo_updates}")
    logger.info(f"处理错误次数: {processing_errors}")
    logger.info(f"成功率: {100 * len(results) / len(nav_frames):.1f}%")

    # 轨迹统计
    if len(results) > 0:
        lat_range = results_df['lat_deg'].max() - results_df['lat_deg'].min()
        lon_range = results_df['lon_deg'].max() - results_df['lon_deg'].min()
        h_range = results_df['h'].max() - results_df['h'].min()

        # 计算轨迹长度（近似）
        lat_mean = results_df['lat_deg'].mean()
        lat_dist = lat_range * 111000  # 1度纬度约111km
        lon_dist = lon_range * 111000 * np.cos(np.radians(lat_mean))  # 经度距离需要考虑纬度

        logger.info(f"=== 轨迹统计 ===")
        logger.info(f"纬度范围: {lat_range:.6f}° ({lat_dist:.1f}m)")
        logger.info(f"经度范围: {lon_range:.6f}° ({lon_dist:.1f}m)")
        logger.info(f"高程范围: {h_range:.2f}m")
        logger.info(f"轨迹包络: {max(lat_dist, lon_dist):.1f}m × {min(lat_dist, lon_dist):.1f}m")

        # 速度统计
        speed_horizontal = np.sqrt(results_df['v_e'] ** 2 + results_df['v_n'] ** 2)
        logger.info(f"水平速度: 平均={speed_horizontal.mean():.2f} m/s, 最大={speed_horizontal.max():.2f} m/s")

        # 姿态统计
        logger.info(f"姿态范围: roll=[{results_df['roll_deg'].min():.1f}°, {results_df['roll_deg'].max():.1f}°], "
                    f"pitch=[{results_df['pitch_deg'].min():.1f}°, {results_df['pitch_deg'].max():.1f}°], "
                    f"yaw=[{results_df['yaw_deg'].min():.1f}°, {results_df['yaw_deg'].max():.1f}°]")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("用户中断程序执行")
    except Exception as e:
        logger.error(f"程序执行过程中发生严重错误: {e}")
        import traceback

        traceback.print_exc()
