"""配置文件加载模块"""
import os
import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """配置加载器"""

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置加载器

        Args:
            config_path: 配置文件路径，如果为None则使用默认路径
        """
        if config_path is None:
            # 默认配置文件路径
            self.config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
        else:
            self.config_path = Path(config_path)

        self.config = {}
        self._load_config()

    def _load_config(self):
        """加载配置文件"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    if self.config_path.suffix in ['.yaml', '.yml']:
                        self.config = yaml.safe_load(f)
                    elif self.config_path.suffix == '.json':
                        self.config = json.load(f)
                    else:
                        raise ValueError(f"不支持的配置文件格式: {self.config_path.suffix}")
                logger.info(f"配置文件加载成功: {self.config_path}")
            else:
                logger.warning(f"配置文件不存在: {self.config_path}，使用默认配置")
                self.config = self._get_default_config()
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            self.config = self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "data_sources": {
                "yahoo_finance": {
                    "enabled": True,
                    "timeout": 30,
                    "retry_count": 3,
                    "cache_days": 7
                }
            },
            "stock_pool": {
                "csi300": True,
                "csi500": True,
                "custom_stocks": []
            },
            "data_frequency": {
                "daily": True,
                "weekly": False,
                "monthly": False
            },
            "backtest": {
                "initial_capital": 1000000,
                "commission_rate": 0.0003,
                "slippage_rate": 0.0001,
                "benchmark": "000300.SH"
            },
            "logging": {
                "level": "INFO",
                "file": "./logs/quant_stock.log"
            }
        }

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值

        Args:
            key: 配置键，支持点分隔符如"data_sources.yahoo_finance.enabled"
            default: 默认值

        Returns:
            配置值或默认值
        """
        keys = key.split('.')
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def set(self, key: str, value: Any):
        """
        设置配置值

        Args:
            key: 配置键，支持点分隔符
            value: 配置值
        """
        keys = key.split('.')
        config = self.config

        # 遍历到最后一个键的父级
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # 设置值
        config[keys[-1]] = value

    def save(self, path: Optional[str] = None):
        """
        保存配置到文件

        Args:
            path: 保存路径，如果为None则使用原始路径
        """
        save_path = Path(path) if path else self.config_path

        # 确保目录存在
        save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                if save_path.suffix in ['.yaml', '.yml']:
                    yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
                elif save_path.suffix == '.json':
                    json.dump(self.config, f, indent=2, ensure_ascii=False)
                else:
                    raise ValueError(f"不支持的配置文件格式: {save_path.suffix}")

            logger.info(f"配置文件保存成功: {save_path}")
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")
            raise

    def reload(self):
        """重新加载配置文件"""
        self._load_config()

    def get_all(self) -> Dict[str, Any]:
        """获取所有配置"""
        return self.config.copy()

    def update(self, new_config: Dict[str, Any]):
        """
        更新配置

        Args:
            new_config: 新的配置字典
        """
        self._deep_update(self.config, new_config)

    def _deep_update(self, original: Dict, update: Dict):
        """深度更新字典"""
        for key, value in update.items():
            if key in original and isinstance(original[key], dict) and isinstance(value, dict):
                self._deep_update(original[key], value)
            else:
                original[key] = value


# 全局配置实例
_config_loader = None


def get_config_loader(config_path: Optional[str] = None) -> ConfigLoader:
    """
    获取配置加载器实例（单例模式）

    Args:
        config_path: 配置文件路径

    Returns:
        ConfigLoader实例
    """
    global _config_loader

    if _config_loader is None:
        _config_loader = ConfigLoader(config_path)

    return _config_loader


def get_config(key: str = None, default: Any = None) -> Any:
    """
    快速获取配置值

    Args:
        key: 配置键，如果为None则返回整个配置字典
        default: 默认值

    Returns:
        配置值或整个配置字典
    """
    config_loader = get_config_loader()
    if key is None:
        return config_loader.get_all()
    return config_loader.get(key, default)


def get_nested_config(config_dict: dict, key: str, default: Any = None) -> Any:
    """
    从嵌套字典中获取配置值（支持点分隔符）

    Args:
        config_dict: 配置字典
        key: 配置键，支持点分隔符
        default: 默认值

    Returns:
        配置值或默认值
    """
    if not config_dict or not key:
        return default

    keys = key.split('.')
    value = config_dict

    try:
        for k in keys:
            value = value[k]
        return value
    except (KeyError, TypeError):
        return default


if __name__ == "__main__":
    # 测试配置加载器
    config_loader = ConfigLoader()

    # 获取配置
    print("数据源配置:", config_loader.get("data_sources.yahoo_finance.enabled"))
    print("回测初始资金:", config_loader.get("backtest.initial_capital"))

    # 设置新配置
    config_loader.set("test.key", "value")
    print("测试配置:", config_loader.get("test.key"))

    # 保存配置
    config_loader.save("test_config.yaml")