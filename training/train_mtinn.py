# multi_source_fusion/training/train_mtinn.py

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import sys
import numpy as np
import pandas as pd
import json

# --- 关键设置：确保脚本能找到项目中的其他模块 ---
# 将项目的根目录添加到Python的模块搜索路径中
# 这使得我们可以使用如 from core.config import config 这样的绝对导入
# 无论我们从哪里运行这个脚本
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
# ----------------------------------------------------

from core.config import config
from data.dataset import MTINNDataset
from data import loaders
from modules.mtinn_model import MTINN
from modules.mtinn_loss import MTINNLoss


def prepare_training_data(imu_path, odo_path, gt_path):
    """
    一个统一的函数，负责加载、同步、计算变化量，并返回可用于训练的Numpy数组。
    这是训练前最核心的数据准备步骤。
    """
    # 1. 加载所有原始数据
    print("--- 开始准备训练数据 ---")
    print("步骤 1/4: 加载原始数据文件...")
    imu_df = loaders.load_imu_data(imu_path)
    odo_df = loaders.load_odo_data(odo_path)

    # 直接使用pandas加载已经转换好的、带表头的真值CSV文件
    try:
        gt_df = pd.read_csv(gt_path, comment='%')
        print(f"成功加载地面真值文件: {gt_path}")
    except FileNotFoundError:
        print(f"错误: 找不到地面真值文件 '{gt_path}'。请确保该文件存在。")
        raise
    except Exception as e:
        print(f"加载地面真值文件时出错: {e}")
        raise

    if imu_df.empty or gt_df.empty:
        raise ValueError("IMU或地面真值数据为空，无法继续。")

    # 2. 高效同步数据
    print("步骤 2/4: 使用Pandas进行时间同步...")
    imu_df.set_index('timestamp', inplace=True)
    gt_df.set_index('timestamp', inplace=True)
    if not odo_df.empty:
        odo_df.set_index('timestamp', inplace=True)

    # 以IMU的时间戳为基准，将所有数据合并到同一个DataFrame
    # 使用左连接(left join)保留所有IMU时间点
    df = imu_df.join(gt_df, how='left')
    if not odo_df.empty:
        df = df.join(odo_df, how='left')

    # 使用线性插值填充所有因频率不同而产生的缺失值
    df.interpolate(method='linear', limit_direction='both', inplace=True)
    df.dropna(inplace=True)  # 删除无法插值的行（通常是开头或结尾）

    if df.empty:
        raise ValueError("数据同步后没有有效的样本。请检查各数据文件的时间戳是否重叠。")

    print(f"同步完成，共得到 {len(df)} 个有效数据点。")

    # 3. 准备模型输入特征
    # 顺序: ax, ay, az, gx, gy, gz, velocity
    input_features = df[['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'velocity']].values.astype(np.float32)

    # 4. 准备训练目标（计算状态变化量）
    print("步骤 3/4: 计算状态变化量作为训练目标...")
    # 定义真值状态的列名
    gt_cols = ['roll', 'pitch', 'yaw', 'v_e', 'v_n', 'v_u', 'lat', 'lon', 'h']

    # 将角度单位从度转换为弧度，以便进行正确的差分计算
    for col in ['roll', 'pitch', 'yaw', 'lat', 'lon']:
        if col in df.columns:
            df[col] = np.radians(df[col])

    # 使用Pandas的 .diff() 方法高效计算相邻行之间的差值
    delta_df = df[gt_cols].diff()
    delta_df.iloc[0] = 0  # 第一行的变化量定义为0

    # **关键处理**：处理角度的环绕问题（例如从359度变到1度，差值应为+2度，而不是-358度）
    for col in ['roll', 'pitch', 'yaw', 'lon']:  # 纬度一般不存在环绕问题
        delta_df[col] = (delta_df[col] + np.pi) % (2 * np.pi) - np.pi

    target_deltas = delta_df.values.astype(np.float32)

    print("步骤 4/4: 数据准备完成！")
    print(f"  - 输入特征形状: {input_features.shape}")
    print(f"  - 目标(变化量)形状: {target_deltas.shape}")
    return input_features, target_deltas


class LossTracker:
    """一个简单的工具类，用于在训练过程中跟踪和保存损失曲线。"""

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.history = []
        print(f"损失记录器已初始化，将保存日志到: {self.save_dir}")

    def add_entry(self, epoch, train_loss, val_loss, lr):
        entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'learning_rate': lr
        }
        self.history.append(entry)

    def save(self):
        # 保存为JSON文件，便于机器读取
        with open(os.path.join(self.save_dir, 'training_history.json'), 'w') as f:
            json.dump(self.history, f, indent=4)

        # 保存为CSV文件，便于用Excel等工具查看
        df = pd.DataFrame(self.history)
        df.to_csv(os.path.join(self.save_dir, 'training_history.csv'), index=False)
        print("训练历史已更新并保存。")


def train(ground_truth_path: str):
    """
    完整的模型训练主函数。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 训练开始 ---")
    print(f"使用设备: {device}")

    # --- 1. 数据准备 ---
    imu_path = config.get('data_paths.imu_path')
    odo_path = config.get('data_paths.odo_path')

    try:
        input_data, target_data = prepare_training_data(imu_path, odo_path, ground_truth_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"错误: 数据准备失败 - {e}")
        return

    # --- 2. 创建数据集和数据加载器 ---
    full_dataset = MTINNDataset(input_data, target_data)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    batch_size = config.get('mtinn_hyperparams.batch_size', 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"数据集划分完成: {len(train_dataset)} 训练样本, {len(val_dataset)} 验证样本。")

    # --- 3. 初始化模型、损失函数和优化器 ---
    model = MTINN().to(device)
    loss_fn = MTINNLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.get('mtinn_hyperparams.learning_rate', 0.001))
    # 使用学习率调度器：当验证损失不再下降时，自动降低学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # --- 4. 初始化训练监控工具 ---
    model_save_dir = config.get('data_paths.model_save_dir', 'models/')
    os.makedirs(model_save_dir, exist_ok=True)
    loss_tracker = LossTracker(model_save_dir)
    best_val_loss = float('inf')

    # --- 5. 训练循环 ---
    num_epochs = config.get('mtinn_hyperparams.num_epochs', 400)
    print(f"\n--- 开始训练循环 (共 {num_epochs} 个 epochs) ---")

    for epoch in range(num_epochs):
        # --- 训练阶段 ---
        print(f"\nEpoch {epoch + 1}/{num_epochs} 开始...")
        model.train()
        print("  -> 模型训练中...")
        total_train_loss = 0.0
        for batch in train_loader:
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)

            optimizer.zero_grad()
            pred_deltas = model(inputs)  # 模型预测的是变化量

            # 计算损失（数据损失+物理损失）
            loss = loss_fn(pred_deltas, inputs, targets, 1.0 / config.get('sensor_params.imu.rate_hz'))

            if torch.isnan(loss):
                print("警告: 训练损失出现NaN，跳过此批次。")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 防止梯度爆炸
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # --- 验证阶段 ---
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(device)
                targets = batch['target'].to(device)
                pred_deltas = model(inputs)
                loss = loss_fn(pred_deltas, inputs, targets, 1.0 / config.get('sensor_params.imu.rate_hz'))
                if not torch.isnan(loss):
                    total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        # 更新学习率
        scheduler.step(avg_val_loss)

        # --- 日志记录与模型保存 ---
        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {avg_val_loss:.6f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.1e}")

        loss_tracker.add_entry(epoch + 1, avg_train_loss, avg_val_loss, optimizer.param_groups[0]['lr'])
        loss_tracker.save()  # 每个epoch都保存一次训练历史

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(model_save_dir, 'best_mtinn_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> 验证损失降低，最佳模型已保存至: {best_model_path}")

    print("\n--- 训练完成 ---")
    print(f"最佳验证损失: {best_val_loss:.6f}")


if __name__ == '__main__':
    # 从配置文件中获取真值文件的路径
    gt_path = config.get('data_paths.ground_truth_path', 'data_raw/truth.nav')

    if not os.path.exists(gt_path):
        print(f"错误: 在config.yaml中指定的地面真值文件未找到: '{gt_path}'")
        sys.exit(1)

    try:
        train(ground_truth_path=gt_path)
    except Exception as e:
        print(f"\n训练过程中发生未处理的严重错误: {e}")
        import traceback

        traceback.print_exc()