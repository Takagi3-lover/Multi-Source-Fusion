# multi_source_fusion/core/config.py

import yaml
from typing import Any
import os


class Config:
    """
    一个单例类，用于加载和管理项目的配置参数。
    在系统启动时从指定的YAML文件加载配置，并提供一个全局访问点。
    """
    _instance = None
    _config_data = None

    def __new__(cls, config_path: str = 'config.yaml'):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config(config_path)
        return cls._instance

    def _load_config(self, config_path: str):
        """加载YAML配置文件。"""
        try:
            # 确保路径存在
            if not os.path.exists(config_path):
                print(f"警告：配置文件 '{config_path}' 未找到，使用默认配置。")
                self._config_data = self._get_default_config()
                return

            with open(config_path, 'r', encoding='utf-8') as f:
                self._config_data = yaml.safe_load(f)
            print(f"配置已从 '{config_path}' 加载。")
        except yaml.YAMLError as e:
            print(f"错误：解析配置文件 '{config_path}' 时出错: {e}")
            self._config_data = self._get_default_config()
        except Exception as e:
            print(f"错误：加载配置文件时出现未知错误: {e}")
            self._config_data = self._get_default_config()

    def _get_default_config(self) -> dict:
        """返回默认配置，防止配置文件缺失时程序崩溃。"""
        return {
            'data_paths': {
                'imu_path': 'data_raw/imu.txt',
                'gnss_path': 'data_raw/gnss.pos',
                'odo_path': 'data_raw/odo.csv',
                'map_path': 'data_raw/track.nav',
                'output_path': 'results/'
            },
            'sensor_params': {
                'imu': {'rate_hz': 200.0},
                'gnss': {'rate_hz': 1.0},
                'odo': {'rate_hz': 200.0}
            },
            'mtinn_hyperparams': {
                'learning_rate': 0.001,
                'batch_size': 64,
                'num_epochs': 100,
                'sequence_length': 50,
                'dropout_rate': 0.2
            },
            'physical_params': {
                'earth_rate': 7.292115e-5,
                'earth_a': 6378137.0,
                'earth_e2': 0.00669438
            },
            'fusion_strategy': {
                'mode': 'math_only',  # 默认使用纯数学模型
                'model_weight': 0.0
            }
        }

    def get(self, key: str, default: Any = None) -> Any:
        """
        通过点分隔的键路径获取配置值。
        例如: get('data_paths.imu_path')
        """
        if self._config_data is None:
            return default

        keys = key.split('.')
        value = self._config_data
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def validate_fusion_strategy(self):
        """验证融合策略配置的合理性"""
        mode = self.get('fusion_strategy.mode', 'math_only')
        model_weight = self.get('fusion_strategy.model_weight', 0.0)

        valid_modes = ['math_only', 'model_only', 'weighted_fusion', 'adaptive_fusion']

        if mode not in valid_modes:
            print(f"警告: 无效的融合模式 '{mode}', 使用默认值 'math_only'")
            self._config_data['fusion_strategy']['mode'] = 'math_only'
            mode = 'math_only'

        if mode == 'weighted_fusion' and not (0.0 <= model_weight <= 1.0):
            print(f"警告: 加权融合模式下模型权重 {model_weight} 无效，设置为 0.1")
            self._config_data['fusion_strategy']['model_weight'] = 0.1

        return mode, self.get('fusion_strategy.model_weight', 0.0)


# 创建一个全局可访问的配置实例
config = Config()

if __name__ == '__main__':
    # 测试配置加载和访问
    print("地球自转速率:", config.get('physical_params.earth_rate'))
    print("IMU采样率:", config.get('sensor_params.imu.rate_hz'))
    print("融合策略:", config.get('fusion_strategy.mode'))
    config.validate_fusion_strategy()