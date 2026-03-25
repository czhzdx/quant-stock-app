"""降级链工具模块"""
import asyncio
import logging
from typing import Callable, List, TypeVar, Optional, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')


class FallbackResult:
    """降级链执行结果"""
    def __init__(self, success: bool, result: Any = None, provider: str = None, error: Exception = None):
        self.success = success
        self.result = result
        self.provider = provider
        self.error = error

    def __bool__(self):
        return self.success


class FallbackChain:
    """
    通用降级链实现

    用于按顺序尝试多个数据源/API，直到成功或全部失败

    Usage:
        chain = FallbackChain([provider_a, provider_b, provider_c])
        result = await chain.execute("600519")
    """

    def __init__(
        self,
        providers: List[Callable],
        names: List[str] = None,
        retry_count: int = 1,
        retry_delay: float = 1.0
    ):
        """
        初始化降级链

        Args:
            providers: 提供者函数列表
            names: 提供者名称列表（用于日志）
            retry_count: 每个提供者的重试次数
            retry_delay: 重试间隔（秒）
        """
        self.providers = providers
        self.names = names or [f"provider_{i}" for i in range(len(providers))]
        self.retry_count = retry_count
        self.retry_delay = retry_delay

        if len(self.providers) != len(self.names):
            raise ValueError("providers和names长度必须一致")

    async def execute(self, *args, **kwargs) -> FallbackResult:
        """
        执行降级链

        按顺序尝试每个提供者，直到成功或全部失败

        Returns:
            FallbackResult: 执行结果
        """
        last_error = None

        for name, provider in zip(self.names, self.providers):
            for attempt in range(self.retry_count):
                try:
                    logger.info(f"尝试 {name} (尝试 {attempt + 1}/{self.retry_count})")

                    # 支持同步和异步函数
                    if asyncio.iscoroutinefunction(provider):
                        result = await provider(*args, **kwargs)
                    else:
                        result = provider(*args, **kwargs)

                    if result is not None:
                        logger.info(f"{name} 执行成功")
                        return FallbackResult(
                            success=True,
                            result=result,
                            provider=name
                        )
                    else:
                        logger.warning(f"{name} 返回空结果")

                except Exception as e:
                    logger.warning(f"{name} 执行失败: {e}")
                    last_error = e

                if attempt < self.retry_count - 1:
                    await asyncio.sleep(self.retry_delay)

        logger.error(f"所有提供者执行失败")
        return FallbackResult(
            success=False,
            error=last_error or Exception("All providers returned None")
        )

    def execute_sync(self, *args, **kwargs) -> FallbackResult:
        """同步执行降级链"""
        return asyncio.run(self.execute(*args, **kwargs))


class DataProvider:
    """数据提供者基类"""

    def __init__(self, name: str, config: dict = None):
        self.name = name
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)

    async def fetch(self, *args, **kwargs) -> Any:
        """获取数据（子类实现）"""
        raise NotImplementedError

    def is_available(self) -> bool:
        """检查提供者是否可用"""
        return self.enabled


def create_fallback_chain_from_config(
    providers_config: dict,
    provider_factory: Callable[[str, dict], DataProvider],
    primary: str = None,
    fallback_order: List[str] = None
) -> FallbackChain:
    """
    从配置创建降级链

    Args:
        providers_config: 提供者配置字典
        provider_factory: 提供者工厂函数
        primary: 主提供者名称
        fallback_order: 降级顺序

    Returns:
        FallbackChain实例
    """
    # 确定顺序
    if primary and fallback_order:
        order = [primary] + fallback_order
    elif primary:
        order = [primary]
    else:
        order = list(providers_config.keys())

    # 创建提供者实例
    providers = []
    names = []

    for name in order:
        if name in providers_config:
            config = providers_config[name]
            provider = provider_factory(name, config)
            if provider.is_available():
                providers.append(provider.fetch)
                names.append(name)

    return FallbackChain(providers, names)


# 便捷函数
async def try_providers(
    providers: List[Callable],
    *args,
    names: List[str] = None,
    **kwargs
) -> FallbackResult:
    """
    快速尝试多个提供者

    Usage:
        result = await try_providers(
            [fetch_from_a, fetch_from_b],
            "600519",
            names=["source_a", "source_b"]
        )
    """
    chain = FallbackChain(providers, names)
    return await chain.execute(*args, **kwargs)
