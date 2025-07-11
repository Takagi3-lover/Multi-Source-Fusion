# multi_source_fusion/inference/predictor.py

import torch
import numpy as np
from typing import Dict, Tuple, Optional

from ..modules.mtinn_model import MTINN
from ..core.types import SystemState


class MTINNPredictor:
    """
    一个围绕已训练MTINN模型的包装器，用于单步预测和不确定性估计。
    """

    def __init__(self, model_path: str, device: torch.device):
        """
        Args:
            model_path (str): 训练好的模型权重文件路径 (.pth)。
            device (torch.device): 运行模型的设备 (cpu or cuda)。
        """
        self.device = device
        self.model = MTINN().to(device)

        try:
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.eval()  # 默认设置为评估模式
            print(f"MTINN预测器已加载模型: {model_path}")
        except FileNotFoundError:
            print(f"错误: 模型文件 {model_path} 未找到")
            raise
        except Exception as e:
            print(f"错误: 加载模型时出错: {e}")
            raise

    def predict_step_with_uncertainty(
            self,
            imu_data: Dict[str, float],
            odo_data: Dict[str, float],
            prev_state: SystemState,
            feedback_error: Optional[np.ndarray] = None,
            n_samples: int = 20
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        使用蒙特卡洛Dropout (MC Dropout) 执行单步预测并估计不确定性。

        Args:
            imu_data (Dict[str, float]): 当前IMU读数。
            odo_data (Dict[str, float]): 当前ODO读数。
            prev_state (SystemState): 上一时刻的系统状态。
            feedback_error (Optional[np.ndarray]): 来自地图匹配的反馈误差。
            n_samples (int): MC Dropout的采样次数。

        Returns:
            Tuple[Dict[str, np.ndarray], np.ndarray]:
                - 预测状态的均值 (字典格式)。
                - 预测的协方差矩阵 Q_k (9x9 numpy数组)。
        """
        # 1. 准备单步输入张量
        input_vec = np.array([
            imu_data['ax'], imu_data['ay'], imu_data['az'],
            imu_data['gx'], imu_data['gy'], imu_data['gz'],
            odo_data['velocity']
        ], dtype=np.float32)

        # 将输入扩展为 (batch_size=n_samples, seq_len=1, input_size=7)
        input_tensor = torch.from_numpy(input_vec).unsqueeze(0).unsqueeze(0).to(self.device)
        input_tensor = input_tensor.repeat(n_samples, 1, 1)

        # 2. 启用Dropout以进行不确定性估计
        self.model.train()

        # 3. 多次前向传播
        predictions_list = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.model(input_tensor[:1])  # 只取一个样本进行预测
                predictions_list.append(pred.squeeze(1))  # 移除seq_len维度

        # 堆叠所有预测结果
        predictions = torch.stack(predictions_list, dim=0)  # (n_samples, 1, 9)
        predictions = predictions.squeeze(1)  # (n_samples, 9)

        # 4. 计算均值和协方差
        mean_prediction = predictions.mean(dim=0).cpu().numpy()

        # 计算协方差矩阵
        predictions_centered = predictions - predictions.mean(dim=0, keepdim=True)
        covariance_prediction = torch.mm(predictions_centered.T, predictions_centered) / (n_samples - 1)
        covariance_prediction = covariance_prediction.cpu().numpy()

        # 5. 格式化输出
        predicted_state_dict = {
            'attitude': mean_prediction[:3],
            'velocity': mean_prediction[3:6],
            'position': mean_prediction[6:9]
        }

        # 将上一时刻的位置和速度加到预测的增量上
        # 注意：MTINN被设计为预测下一个完整状态，而不是增量。
        # 如果它被训练为预测增量，则需要加上 prev_state。
        # 根据设计文档，MTINN直接预测状态 Yt，所以这里不需要加法。

        # 如果有反馈误差，则应用它
        if feedback_error is not None and len(feedback_error) >= 3:
            predicted_state_dict['position'] += feedback_error[:3]

        # 恢复评估模式
        self.model.eval()

        return predicted_state_dict, covariance_prediction