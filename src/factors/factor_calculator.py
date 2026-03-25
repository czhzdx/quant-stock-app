"""因子计算模块"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
import talib
from ..utils.config_loader import get_config

logger = logging.getLogger(__name__)


class FactorType(Enum):
    """因子类型枚举"""
    TECHNICAL = "technical"      # 技术指标
    FUNDAMENTAL = "fundamental"  # 基本面因子
    VOLUME_PRICE = "volume_price" # 量价因子
    MARKET = "market"           # 市场因子
    CUSTOM = "custom"           # 自定义因子


@dataclass
class FactorDefinition:
    """因子定义"""
    name: str                    # 因子名称
    factor_type: FactorType      # 因子类型
    description: str            # 因子描述
    parameters: Dict            # 参数
    calculation_func: callable  # 计算函数


class FactorCalculator:
    """因子计算器"""

    def __init__(self):
        """初始化因子计算器"""
        self.config = get_config()  # 获取整个配置字典
        self.factors_config = self.config.get("factors", {})
        self._register_factors()

    def _register_factors(self):
        """注册所有因子"""
        self.factors = {}

        # 注册技术指标因子
        self._register_technical_factors()

        # 注册基本面因子
        self._register_fundamental_factors()

        # 注册量价因子
        self._register_volume_price_factors()

        # 注册市场因子
        self._register_market_factors()

    def _register_technical_factors(self):
        """注册技术指标因子"""
        # MA移动平均线
        ma_periods = self.factors_config.get("technical", {}).get("ma", [5, 10, 20, 60, 120, 250])
        for period in ma_periods:
            self.factors[f"ma_{period}"] = FactorDefinition(
                name=f"ma_{period}",
                factor_type=FactorType.TECHNICAL,
                description=f"{period}日移动平均线",
                parameters={"period": period},
                calculation_func=lambda data, period=period: self._calculate_ma(data, period)
            )

        # MACD
        if self.factors_config.get("technical", {}).get("macd", True):
            self.factors["macd"] = FactorDefinition(
                name="macd",
                factor_type=FactorType.TECHNICAL,
                description="MACD指标",
                parameters={"fast": 12, "slow": 26, "signal": 9},
                calculation_func=self._calculate_macd
            )

        # RSI
        rsi_periods = self.factors_config.get("technical", {}).get("rsi", [6, 12, 24])
        for period in rsi_periods:
            self.factors[f"rsi_{period}"] = FactorDefinition(
                name=f"rsi_{period}",
                factor_type=FactorType.TECHNICAL,
                description=f"{period}日RSI指标",
                parameters={"period": period},
                calculation_func=lambda data, period=period: self._calculate_rsi(data, period)
            )

        # 布林带
        if self.factors_config.get("technical", {}).get("bollinger", True):
            self.factors["bollinger_upper"] = FactorDefinition(
                name="bollinger_upper",
                factor_type=FactorType.TECHNICAL,
                description="布林带上轨",
                parameters={"period": 20, "std": 2},
                calculation_func=lambda data: self._calculate_bollinger(data)[0]
            )
            self.factors["bollinger_middle"] = FactorDefinition(
                name="bollinger_middle",
                factor_type=FactorType.TECHNICAL,
                description="布林带中轨",
                parameters={"period": 20, "std": 2},
                calculation_func=lambda data: self._calculate_bollinger(data)[1]
            )
            self.factors["bollinger_lower"] = FactorDefinition(
                name="bollinger_lower",
                factor_type=FactorType.TECHNICAL,
                description="布林带下轨",
                parameters={"period": 20, "std": 2},
                calculation_func=lambda data: self._calculate_bollinger(data)[2]
            )
            self.factors["bollinger_width"] = FactorDefinition(
                name="bollinger_width",
                factor_type=FactorType.TECHNICAL,
                description="布林带宽度",
                parameters={"period": 20, "std": 2},
                calculation_func=lambda data: self._calculate_bollinger_width(data)
            )

    def _register_fundamental_factors(self):
        """注册基本面因子"""
        fundamental_config = self.factors_config.get("fundamental", {})

        if fundamental_config.get("pe", True):
            self.factors["pe_ratio"] = FactorDefinition(
                name="pe_ratio",
                factor_type=FactorType.FUNDAMENTAL,
                description="市盈率",
                parameters={},
                calculation_func=self._calculate_pe_ratio
            )

        if fundamental_config.get("pb", True):
            self.factors["pb_ratio"] = FactorDefinition(
                name="pb_ratio",
                factor_type=FactorType.FUNDAMENTAL,
                description="市净率",
                parameters={},
                calculation_func=self._calculate_pb_ratio
            )

        if fundamental_config.get("roe", True):
            self.factors["roe"] = FactorDefinition(
                name="roe",
                factor_type=FactorType.FUNDAMENTAL,
                description="净资产收益率",
                parameters={},
                calculation_func=self._calculate_roe
            )

    def _register_volume_price_factors(self):
        """注册量价因子"""
        volume_price_config = self.factors_config.get("volume_price", {})

        if volume_price_config.get("volume_ratio", True):
            self.factors["volume_ratio"] = FactorDefinition(
                name="volume_ratio",
                factor_type=FactorType.VOLUME_PRICE,
                description="成交量比率",
                parameters={"period": 20},
                calculation_func=self._calculate_volume_ratio
            )

        if volume_price_config.get("turnover_rate", True):
            self.factors["turnover_rate"] = FactorDefinition(
                name="turnover_rate",
                factor_type=FactorType.VOLUME_PRICE,
                description="换手率",
                parameters={},
                calculation_func=self._calculate_turnover_rate
            )

        if volume_price_config.get("money_flow", True):
            self.factors["money_flow"] = FactorDefinition(
                name="money_flow",
                factor_type=FactorType.VOLUME_PRICE,
                description="资金流向",
                parameters={"period": 20},
                calculation_func=self._calculate_money_flow
            )

    def _register_market_factors(self):
        """注册市场因子"""
        market_config = self.factors_config.get("market", {})

        if market_config.get("beta", True):
            self.factors["beta"] = FactorDefinition(
                name="beta",
                factor_type=FactorType.MARKET,
                description="Beta系数",
                parameters={"period": 60},
                calculation_func=self._calculate_beta
            )

        if market_config.get("volatility", True):
            self.factors["volatility"] = FactorDefinition(
                name="volatility",
                factor_type=FactorType.MARKET,
                description="波动率",
                parameters={"period": 20},
                calculation_func=self._calculate_volatility
            )

    def calculate_factor(self, data: pd.DataFrame, factor_name: str) -> pd.Series:
        """
        计算单个因子

        Args:
            data: 股票数据DataFrame
            factor_name: 因子名称

        Returns:
            因子值序列
        """
        if factor_name not in self.factors:
            raise ValueError(f"未知的因子: {factor_name}")

        factor_def = self.factors[factor_name]
        logger.info(f"计算因子: {factor_name}")

        try:
            factor_values = factor_def.calculation_func(data)
            factor_values.name = factor_name
            return factor_values
        except Exception as e:
            logger.error(f"计算因子 {factor_name} 失败: {e}")
            return pd.Series(index=data.index, name=factor_name, dtype=float)

    def calculate_multiple_factors(
        self,
        data: pd.DataFrame,
        factor_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        计算多个因子

        Args:
            data: 股票数据DataFrame
            factor_names: 因子名称列表，如果为None则计算所有已注册因子

        Returns:
            包含所有因子值的DataFrame
        """
        if factor_names is None:
            factor_names = list(self.factors.keys())

        factor_data = pd.DataFrame(index=data.index)

        for factor_name in factor_names:
            try:
                factor_series = self.calculate_factor(data, factor_name)
                factor_data[factor_name] = factor_series
            except Exception as e:
                logger.error(f"计算因子 {factor_name} 失败: {e}")
                factor_data[factor_name] = np.nan

        return factor_data

    def calculate_all_factors(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算所有已注册因子"""
        return self.calculate_multiple_factors(data)

    def register_custom_factor(
        self,
        name: str,
        description: str,
        calculation_func: callable,
        parameters: Optional[Dict] = None
    ):
        """
        注册自定义因子

        Args:
            name: 因子名称
            description: 因子描述
            calculation_func: 计算函数
            parameters: 参数字典
        """
        if name in self.factors:
            logger.warning(f"因子 {name} 已存在，将被覆盖")

        self.factors[name] = FactorDefinition(
            name=name,
            factor_type=FactorType.CUSTOM,
            description=description,
            parameters=parameters or {},
            calculation_func=calculation_func
        )

        logger.info(f"已注册自定义因子: {name}")

    # ====== 技术指标计算方法 ======

    def _calculate_ma(self, data: pd.DataFrame, period: int) -> pd.Series:
        """计算移动平均线"""
        close = data['Close']
        return close.rolling(window=period).mean()

    def _calculate_macd(self, data: pd.DataFrame) -> pd.Series:
        """计算MACD"""
        close = data['Close']
        macd_line, signal_line, _ = talib.MACD(
            close.values,
            fastperiod=12,
            slowperiod=26,
            signalperiod=9
        )
        return pd.Series(macd_line - signal_line, index=data.index)

    def _calculate_rsi(self, data: pd.DataFrame, period: int) -> pd.Series:
        """计算RSI"""
        close = data['Close']
        rsi = talib.RSI(close.values, timeperiod=period)
        return pd.Series(rsi, index=data.index)

    def _calculate_bollinger(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算布林带"""
        close = data['Close']
        upper, middle, lower = talib.BBANDS(
            close.values,
            timeperiod=20,
            nbdevup=2,
            nbdevdn=2,
            matype=0
        )
        return (
            pd.Series(upper, index=data.index),
            pd.Series(middle, index=data.index),
            pd.Series(lower, index=data.index)
        )

    def _calculate_bollinger_width(self, data: pd.DataFrame) -> pd.Series:
        """计算布林带宽度"""
        upper, _, lower = self._calculate_bollinger(data)
        return (upper - lower) / ((upper + lower) / 2)

    # ====== 基本面因子计算方法 ======

    def _calculate_pe_ratio(self, data: pd.DataFrame) -> pd.Series:
        """计算市盈率（简化版本，实际需要基本面数据）"""
        # 这里返回一个示例序列，实际应用中需要从基本面数据获取
        return pd.Series(np.nan, index=data.index)

    def _calculate_pb_ratio(self, data: pd.DataFrame) -> pd.Series:
        """计算市净率（简化版本）"""
        return pd.Series(np.nan, index=data.index)

    def _calculate_roe(self, data: pd.DataFrame) -> pd.Series:
        """计算净资产收益率（简化版本）"""
        return pd.Series(np.nan, index=data.index)

    # ====== 量价因子计算方法 ======

    def _calculate_volume_ratio(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """计算成交量比率"""
        volume = data['Volume']
        volume_ma = volume.rolling(window=period).mean()
        return volume / volume_ma

    def _calculate_turnover_rate(self, data: pd.DataFrame) -> pd.Series:
        """计算换手率（简化版本）"""
        # 需要流通股本数据
        return pd.Series(np.nan, index=data.index)

    def _calculate_money_flow(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """计算资金流向"""
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        money_flow = typical_price * data['Volume']

        # 计算资金流向指标
        money_flow_positive = money_flow.where(data['Close'] > data['Close'].shift(1), 0)
        money_flow_negative = money_flow.where(data['Close'] < data['Close'].shift(1), 0)

        mf_ratio = money_flow_positive.rolling(window=period).sum() / \
                   money_flow_negative.rolling(window=period).sum()

        return mf_ratio

    # ====== 市场因子计算方法 ======

    def _calculate_beta(self, data: pd.DataFrame, period: int = 60) -> pd.Series:
        """计算Beta系数（需要市场指数数据）"""
        # 这里返回一个示例序列，实际应用中需要市场指数数据
        return pd.Series(np.nan, index=data.index)

    def _calculate_volatility(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """计算波动率"""
        returns = data['Close'].pct_change()
        volatility = returns.rolling(window=period).std() * np.sqrt(252)  # 年化波动率
        return volatility

    # ====== 辅助方法 ======

    def get_factor_info(self, factor_name: str) -> Optional[Dict]:
        """获取因子信息"""
        if factor_name in self.factors:
            factor_def = self.factors[factor_name]
            return {
                "name": factor_def.name,
                "type": factor_def.factor_type.value,
                "description": factor_def.description,
                "parameters": factor_def.parameters
            }
        return None

    def list_factors(self, factor_type: Optional[FactorType] = None) -> List[str]:
        """列出所有因子"""
        if factor_type is None:
            return list(self.factors.keys())
        else:
            return [name for name, factor in self.factors.items()
                    if factor.factor_type == factor_type]

    def describe_factors(self) -> pd.DataFrame:
        """描述所有因子"""
        factor_list = []
        for name, factor in self.factors.items():
            factor_list.append({
                "name": name,
                "type": factor.factor_type.value,
                "description": factor.description,
                "parameters": factor.parameters
            })

        return pd.DataFrame(factor_list)


# 测试函数
def test_factor_calculator():
    """测试因子计算器"""
    import logging
    logging.basicConfig(level=logging.INFO)

    # 创建示例数据
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)

    data = pd.DataFrame({
        'Open': np.random.randn(100).cumsum() + 100,
        'High': np.random.randn(100).cumsum() + 105,
        'Low': np.random.randn(100).cumsum() + 95,
        'Close': np.random.randn(100).cumsum() + 100,
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)

    # 初始化因子计算器
    calculator = FactorCalculator()

    # 测试计算单个因子
    print("测试计算移动平均线...")
    ma_20 = calculator.calculate_factor(data, "ma_20")
    print(f"MA20前5个值:\n{ma_20.head()}")

    # 测试计算多个因子
    print("\n测试计算多个因子...")
    factor_names = ["ma_10", "rsi_14", "bollinger_width"]
    factors_df = calculator.calculate_multiple_factors(data, factor_names)
    print(f"因子数据形状: {factors_df.shape}")
    print(f"因子数据前5行:\n{factors_df.head()}")

    # 测试计算所有因子
    print("\n测试计算所有因子...")
    all_factors = calculator.calculate_all_factors(data)
    print(f"所有因子数据形状: {all_factors.shape}")

    # 测试列出因子
    print("\n测试列出技术指标因子...")
    technical_factors = calculator.list_factors(FactorType.TECHNICAL)
    print(f"技术指标因子: {technical_factors}")

    # 测试获取因子信息
    print("\n测试获取因子信息...")
    factor_info = calculator.get_factor_info("ma_20")
    print(f"MA20因子信息: {factor_info}")

    # 测试描述所有因子
    print("\n测试描述所有因子...")
    factor_desc = calculator.describe_factors()
    print(f"因子描述:\n{factor_desc}")

    # 测试注册自定义因子
    print("\n测试注册自定义因子...")

    def custom_momentum(data: pd.DataFrame, period: int = 20) -> pd.Series:
        """自定义动量因子"""
        returns = data['Close'].pct_change(periods=period)
        return returns

    calculator.register_custom_factor(
        name="momentum_20",
        description="20日动量因子",
        calculation_func=lambda data: custom_momentum(data, 20),
        parameters={"period": 20}
    )

    # 测试计算自定义因子
    momentum = calculator.calculate_factor(data, "momentum_20")
    print(f"动量因子前5个值:\n{momentum.head()}")


if __name__ == "__main__":
    test_factor_calculator()