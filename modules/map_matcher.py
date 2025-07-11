# multi_source_fusion/modules/map_matcher.py

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from typing import Optional

from ..core.types import SystemState
from ..core.config import config
from ..core.coordinates import CoordinateSystem


class MapMatcher:
    """
    将融合后的定位结果匹配到已知的铁路线路地图上。
    """

    def __init__(self, map_df: pd.DataFrame):
        """
        初始化地图匹配器。

        Args:
            map_df (pd.DataFrame): 地图数据DataFrame。
        """
        if map_df.empty:
            raise ValueError("地图数据为空，无法初始化MapMatcher。")

        required_cols = ['lat', 'lon', 'h']
        missing_cols = [col for col in required_cols if col not in map_df.columns]
        if missing_cols:
            raise ValueError(f"地图数据缺少必要列: {missing_cols}")

        self.map_df = map_df.copy()

        # 1. 将地图点的经纬高转换为ECEF坐标，用于构建KDTree
        self.map_ecef = np.array([
            CoordinateSystem.wgs84_to_ecef(row['lat'], row['lon'], row['h'])
            for _, row in map_df.iterrows()
        ])

        # 2. 构建KDTree以便快速搜索最近邻
        try:
            self.kdtree = KDTree(self.map_ecef)
        except Exception as e:
            print(f"构建KDTree失败: {e}")
            raise

        # 3. 从配置加载参数
        self.search_radius = config.get('map_matching.search_radius_m', 50.0)
        self.weight_dist = config.get('map_matching.weight_dist', 0.6)
        self.weight_angle = config.get('map_matching.weight_angle', 0.4)

        print(f"地图匹配器初始化完成，加载了 {len(map_df)} 个地图点")

    def match(self, current_state: SystemState) -> SystemState:
        """
        执行地图匹配。

        Args:
            current_state (SystemState): EKF输出的当前状态。

        Returns:
            SystemState: 匹配到地图上的新状态。
        """
        try:
            # 1. 将当前位置转换为ECEF以查询KDTree
            current_pos_rad = current_state.position
            current_ecef = CoordinateSystem.wgs84_to_ecef(
                current_pos_rad[0], current_pos_rad[1], current_pos_rad[2]
            )

            # 2. 搜索候选点
            # 查询KDTree以找到在搜索半径内的所有地图点的索引
            candidate_indices = self.kdtree.query_ball_point(current_ecef, r=self.search_radius)

            if not candidate_indices:
                # 如果没有找到候选点，则返回原始状态
                print("警告: 未找到候选地图点，返回原始状态")
                return current_state

            # 3. 计算每个候选点的得分
            best_score = -1
            best_match_idx = -1

            # 计算当前车辆的行驶方向 (ENU)
            v_e, v_n = current_state.velocity[0], current_state.velocity[1]
            current_azimuth = np.arctan2(v_e, v_n)  # ENU中的方位角

            for idx in candidate_indices:
                # 候选地图点信息
                candidate_point = self.map_df.iloc[idx]
                candidate_ecef = self.map_ecef[idx]

                # a. 距离得分
                dist_3d = np.linalg.norm(current_ecef - candidate_ecef)
                score_dist = max(0.0, 1.0 - (dist_3d / self.search_radius))  # 距离越近，得分越高

                # b. 方向得分
                track_azimuth = np.radians(candidate_point.get('azimuth', 0))
                angle_diff = abs(current_azimuth - track_azimuth)
                # 处理角度环绕
                if angle_diff > np.pi:
                    angle_diff = 2 * np.pi - angle_diff
                score_angle = max(0.0, 1.0 - (angle_diff / np.pi))  # 角度差越小，得分越高

                # c. 总分
                total_score = self.weight_dist * score_dist + self.weight_angle * score_angle

                if total_score > best_score:
                    best_score = total_score
                    best_match_idx = idx

            if best_match_idx == -1:
                print("警告: 未找到最佳匹配点，返回原始状态")
                return current_state

            # 4. 创建匹配后的状态
            matched_point_info = self.map_df.iloc[best_match_idx]

            matched_state = SystemState(
                timestamp=current_state.timestamp,
                attitude=current_state.attitude.copy(),  # 姿态和速度暂时保持不变
                velocity=current_state.velocity.copy(),
                position=np.array([
                    matched_point_info['lat'],
                    matched_point_info['lon'],
                    matched_point_info['h']
                ]),
                covariance=current_state.covariance.copy()  # 协方差也保持不变
            )

            return matched_state

        except Exception as e:
            print(f"地图匹配过程出错: {e}")
            return current_state