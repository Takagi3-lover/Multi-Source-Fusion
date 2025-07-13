# multi_source_fusion/data/dataset.py

import torch
import numpy as np
from torch.utils.data import Dataset
from core.config import config
from typing import List


class MTINNDataset(Dataset):
    """
    一个简洁的PyTorch Dataset。
    它接收已经处理好的、作为Numpy数组的输入和目标数据。
    """

    def __init__(self, input_data: np.ndarray, target_data: np.ndarray):
        """
        Args:
            input_data (np.ndarray): 输入特征，形状 (num_samples, 7)。
            target_data (np.ndarray): 训练目标（变化量），形状 (num_samples, 9)。
        """
        if not isinstance(input_data, np.ndarray) or not isinstance(target_data, np.ndarray):
            raise TypeError("输入和目标数据必须是Numpy数组。")

        if len(input_data) != len(target_data):
            raise ValueError("输入和目标数据的样本数量必须一致。")

        self.input_data = torch.from_numpy(input_data.astype(np.float32))
        self.target_data = torch.from_numpy(target_data.astype(np.float32))
        self.seq_len = config.get('mtinn_hyperparams.sequence_length', 50)

        print(f"MTINNDataset 初始化完成，序列长度: {self.seq_len}")

    def __len__(self):
        # 长度是可生成的独立序列的数量
        return max(0, self.input_data.shape[0] - self.seq_len + 1)

    def __getitem__(self, idx: int):
        """
        返回一个序列的输入数据和对应的目标（变化量）。
        """
        if idx >= len(self):
            raise IndexError(f"索引 {idx} 超出范围")

        # 从张量中切片，效率更高
        input_seq = self.input_data[idx: idx + self.seq_len]
        target_seq = self.target_data[idx: idx + self.seq_len]

        return {'input': input_seq, 'target': target_seq}