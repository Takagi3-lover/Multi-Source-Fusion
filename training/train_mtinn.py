# multi_source_fusion/training/train_mtinn.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import numpy as np
from typing import Optional

from ..core.config import config
from ..data.dataset import create_synchronized_dataset, MTINNDataset
from ..modules.mtinn_model import MTINN
from ..modules.mtinn_loss import MTINNLoss


def train(ground_truth_path: Optional[str] = None):
    """
    主训练函数，用于训练MTINN模型。

    Args:
        ground_truth_path (Optional[str]): 地面真值数据路径。
    """
    # 1. 加载配置并设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 2. 加载和准备数据
    imu_path = config.get('data_paths.imu_path')
    gnss_path = config.get('data_paths.gnss_path')
    odo_path = config.get('data_paths.odo_path')

    try:
        nav_frames = create_synchronized_dataset(imu_path, gnss_path, odo_path)
        print(f"成功加载 {len(nav_frames)} 个导航帧")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    # 创建PyTorch数据集
    # 在真实场景中，第二个参数应该是包含真实GT的列表
    # 这里为了演示，我们使用nav_frames作为输入和伪真值
    try:
        full_dataset = MTINNDataset(nav_frames, ground_truth_frames=nav_frames)
        print(f"数据集创建成功，包含 {len(full_dataset)} 个序列")
    except Exception as e:
        print(f"数据集创建失败: {e}")
        return

    if len(full_dataset) == 0:
        print("错误: 数据集为空，无法训练")
        return

    # 划分训练集和验证集
    train_size = max(1, int(0.8 * len(full_dataset)))
    val_size = len(full_dataset) - train_size

    if val_size == 0:
        val_size = 1
        train_size = len(full_dataset) - 1

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    batch_size = config.get('mtinn_hyperparams.batch_size', 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 3. 初始化模型、损失函数和优化器
    try:
        model = MTINN().to(device)
        loss_fn = MTINNLoss(adaptive_weights=True).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config.get('mtinn_hyperparams.learning_rate', 0.001))
        print("模型、损失函数和优化器初始化成功")
    except Exception as e:
        print(f"模型初始化失败: {e}")
        return

    # 4. 训练循环
    num_epochs = config.get('mtinn_hyperparams.num_epochs', 100)
    best_val_loss = float('inf')

    # 创建模型保存目录
    model_save_dir = config.get('data_paths.output_path', 'results/') + 'models/'
    os.makedirs(model_save_dir, exist_ok=True)

    print(f"开始训练，共 {num_epochs} 个epoch")

    for epoch in range(num_epochs):
        # --- 训练阶段 ---
        model.train()
        total_train_loss = 0
        train_batches = 0

        try:
            for batch_idx, batch in enumerate(train_loader):
                inputs = batch['input'].to(device)

                # 在真实场景中，targets应来自 batch['target']
                # 此处为演示，我们从输入数据中构造伪目标
                # 注意：这里需要确保目标的形状与模型输出匹配
                batch_size, seq_len, input_dim = inputs.shape
                targets = torch.randn(batch_size, seq_len, 9, device=device)  # 伪目标，形状需匹配输出

                optimizer.zero_grad()

                try:
                    predictions = model(inputs)
                    dt = 1.0 / config.get('sensor_params.imu.rate_hz', 100.0)
                    loss = loss_fn(predictions, inputs, dt, targets)

                    loss.backward()

                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()

                    total_train_loss += loss.item()
                    train_batches += 1

                except Exception as e:
                    print(f"训练批次 {batch_idx} 出错: {e}")
                    continue

            if train_batches > 0:
                avg_train_loss = total_train_loss / train_batches
            else:
                print(f"Epoch {epoch + 1}: 没有成功的训练批次")
                continue

        except Exception as e:
            print(f"训练阶段出错: {e}")
            continue

        # --- 验证阶段 ---
        model.eval()
        total_val_loss = 0
        val_batches = 0

        try:
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    inputs = batch['input'].to(device)
                    batch_size, seq_len, input_dim = inputs.shape
                    targets = torch.randn(batch_size, seq_len, 9, device=device)  # 伪目标

                    try:
                        predictions = model(inputs)
                        dt = 1.0 / config.get('sensor_params.imu.rate_hz', 100.0)
                        loss = loss_fn(predictions, inputs, dt, targets)
                        total_val_loss += loss.item()
                        val_batches += 1
                    except Exception as e:
                        print(f"验证批次 {batch_idx} 出错: {e}")
                        continue

            if val_batches > 0:
                avg_val_loss = total_val_loss / val_batches
            else:
                print(f"Epoch {epoch + 1}: 没有成功的验证批次")
                avg_val_loss = float('inf')

        except Exception as e:
            print(f"验证阶段出错: {e}")
            avg_val_loss = float('inf')

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(model_save_dir, 'best_mtinn_model.pth')
            try:
                torch.save(model.state_dict(), model_path)
                print(f"最佳模型已保存至: {model_path}")
            except Exception as e:
                print(f"保存模型失败: {e}")

    print("训练完成")


if __name__ == '__main__':
    print("开始训练MTINN模型...")
    try:
        train()
    except Exception as e:
        print(f"训练过程出现错误: {e}")
    print("训练脚本执行完毕。")