# multi_source_fusion/inference/predictor.py

import torch
import numpy as np
from typing import Dict, Tuple, Optional

from modules.mtinn_model import MTINN
from core.types import SystemState


class MTINNPredictor:
    """
    MTINN预测器，输出状态变化量。
    """

    def __init__(self, model_path: str, device: torch.device):
        self.device = device
        self.model = MTINN().to(device)

        try:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.eval()
            print(f"MTINN预测器已加载模型: {model_path}")
            print("预测器模式: 输出状态变化量")
        except FileNotFoundError:
            print(f"错误: 模型文件 {model_path} 未找到")
            raise
        except Exception as e:
            print(f"错误: 加载模型时出错: {e}")
            raise

    def predict_step_with_uncertainty(
            self,
            imu_data: Dict[str, float],
            odo_data: Optional[Dict[str, float]],  # 改为Optional
            prev_state: SystemState,
            feedback_error: Optional[np.ndarray] = None,
            n_samples: int = 20
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        预测状态变化量并估计不确定性。
        """
        # 1. 准备输入
        # 如果没有里程计数据，使用0作为默认值
        odo_velocity = odo_data['velocity'] if odo_data is not None else 0.0

        input_vec = np.array([
            imu_data['ax'], imu_data['ay'], imu_data['az'],
            imu_data['gx'], imu_data['gy'], imu_data['gz'],
            odo_velocity
        ], dtype=np.float32)

        input_tensor = torch.from_numpy(input_vec).unsqueeze(0).unsqueeze(0).to(self.device)

        # 2. MC Dropout预测
        self.model.train()
        predictions_list = []

        with torch.no_grad():
            for _ in range(n_samples):
                try:
                    pred = self.model(input_tensor)  # (1, 1, 9)
                    predictions_list.append(pred.squeeze())  # (9,)
                except Exception as e:
                    print(f"预测过程中出错: {e}")
                    predictions_list.append(torch.zeros(9, device=self.device))

        if not predictions_list:
            print("警告: 所有预测都失败，返回零变化量")
            delta_state_dict = {
                'attitude': np.zeros(3),
                'velocity': np.zeros(3),
                'position': np.zeros(3)
            }
            covariance_prediction = np.eye(9) * 1e-6
            return delta_state_dict, covariance_prediction

        # 3. 计算统计量
        predictions = torch.stack(predictions_list, dim=0)  # (n_samples, 9)
        mean_delta = predictions.mean(dim=0).cpu().numpy()

        if n_samples > 1:
            predictions_centered = predictions - predictions.mean(dim=0, keepdim=True)
            covariance_prediction = torch.mm(predictions_centered.T, predictions_centered) / (n_samples - 1)
            covariance_prediction = covariance_prediction.cpu().numpy()
        else:
            covariance_prediction = np.eye(9) * 1e-6

        # 确保协方差矩阵正定
        try:
            np.linalg.cholesky(covariance_prediction)
        except np.linalg.LinAlgError:
            covariance_prediction += np.eye(9) * 1e-8

        # 4. 格式化输出 - 这些是变化量
        delta_state_dict = {
            'attitude': mean_delta[:3],
            'velocity': mean_delta[3:6],
            'position': mean_delta[6:9]
        }

        # 5. 应用反馈误差到位置变化量
        if feedback_error is not None and len(feedback_error) >= 3:
            delta_state_dict['position'] += feedback_error[:3] * 0.1  # 小幅调整

        self.model.eval()
        return delta_state_dict, covariance_prediction

    def predict_single_step(
            self,
            imu_data: Dict[str, float],
            odo_data: Optional[Dict[str, float]],  # 改为Optional
            prev_state: SystemState,
            feedback_error: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """执行单步预测，返回状态变化量"""
        # 如果没有里程计数据，使用0作为默认值
        odo_velocity = odo_data['velocity'] if odo_data is not None else 0.0

        input_vec = np.array([
            imu_data['ax'], imu_data['ay'], imu_data['az'],
            imu_data['gx'], imu_data['gy'], imu_data['gz'],
            odo_velocity
        ], dtype=np.float32)

        input_tensor = torch.from_numpy(input_vec).unsqueeze(0).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            try:
                pred = self.model(input_tensor)
                delta_prediction = pred.squeeze().cpu().numpy()
            except Exception as e:
                print(f"单步预测出错: {e}")
                delta_prediction = np.zeros(9)

        delta_state_dict = {
            'attitude': delta_prediction[:3],
            'velocity': delta_prediction[3:6],
            'position': delta_prediction[6:9]
        }

        if feedback_error is not None and len(feedback_error) >= 3:
            delta_state_dict['position'] += feedback_error[:3] * 0.1

        return delta_state_dict