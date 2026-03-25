"""动量策略模块"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from .base import BaseStrategy, SignalType

logger = logging.getLogger(__name__)


class MomentumStrategy(BaseStrategy):
    """动量策略"""

    def __init__(
        self,
        lookback_period: int = 20,
        holding_period: int = 10,
        top_n: int = 10,
        **kwargs
    ):
        """
        初始化动量策略

        Args:
            lookback_period: 回顾期（计算动量的周期）
            holding_period: 持有期
            top_n: 选择前N只股票
        """
        params = {
            "lookback_period": lookback_period,
            "holding_period": holding_period,
            "top_n": top_n
        }
        params.update(kwargs)
        super().__init__(name="Momentum", params=params)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        生成动量策略信号

        Args:
            data: 包含多只股票数据的DataFrame，索引为日期，列为股票代码

        Returns:
            信号序列（针对每只股票）
        """
        if data.empty:
            logger.warning("输入数据为空")
            return pd.Series(dtype=float)

        # 确保数据按日期排序
        data = data.sort_index()

        lookback = self.params["lookback_period"]
        holding = self.params["holding_period"]
        top_n = self.params["top_n"]

        # 计算动量（过去lookback期的收益率）
        momentum = data.pct_change(periods=lookback)

        # 初始化信号序列
        signals = pd.DataFrame(0, index=data.index, columns=data.columns)

        # 生成买入信号
        for i in range(lookback, len(data) - holding):
            current_date = data.index[i]

            # 获取当前动量排名
            current_momentum = momentum.iloc[i]
            ranked_stocks = current_momentum.dropna().sort_values(ascending=False)

            # 选择前top_n只股票
            buy_stocks = ranked_stocks.head(top_n).index

            # 标记买入信号
            for stock in buy_stocks:
                # 在持有期内标记为持有
                for j in range(holding):
                    if i + j < len(signals):
                        signals.loc[data.index[i + j], stock] = SignalType.BUY.value

        return signals

    def generate_signals_single(self, data: pd.Series) -> pd.Series:
        """
        生成单只股票的动量信号（简化版）

        Args:
            data: 单只股票的价格序列

        Returns:
            信号序列
        """
        if len(data) < self.params["lookback_period"] + 1:
            logger.warning("数据长度不足，无法计算动量")
            return pd.Series(SignalType.HOLD.value, index=data.index)

        # 计算动量
        momentum = data.pct_change(periods=self.params["lookback_period"])

        # 初始化信号
        signals = pd.Series(SignalType.HOLD.value, index=data.index)

        # 生成信号
        buy_signals = momentum > 0  # 正动量买入
        sell_signals = momentum < 0  # 负动量卖出

        signals[buy_signals] = SignalType.BUY.value
        signals[sell_signals] = SignalType.SELL.value

        # 确保有足够的计算周期
        signals.iloc[:self.params["lookback_period"]] = SignalType.HOLD.value

        return signals


class DualMomentumStrategy(BaseStrategy):
    """双动量策略（绝对动量 + 相对动量）"""

    def __init__(
        self,
        absolute_lookback: int = 20,
        relative_lookback: int = 60,
        top_n: int = 5,
        **kwargs
    ):
        """
        初始化双动量策略

        Args:
            absolute_lookback: 绝对动量回顾期
            relative_lookback: 相对动量回顾期
            top_n: 选择前N只股票
        """
        params = {
            "absolute_lookback": absolute_lookback,
            "relative_lookback": relative_lookback,
            "top_n": top_n
        }
        params.update(kwargs)
        super().__init__(name="DualMomentum", params=params)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        生成双动量策略信号

        Args:
            data: 包含多只股票数据的DataFrame

        Returns:
            信号序列
        """
        if data.empty:
            logger.warning("输入数据为空")
            return pd.Series(dtype=float)

        data = data.sort_index()

        abs_lookback = self.params["absolute_lookback"]
        rel_lookback = self.params["relative_lookback"]
        top_n = self.params["top_n"]

        # 计算绝对动量（自身收益率）
        absolute_momentum = data.pct_change(periods=abs_lookback)

        # 计算相对动量（相对于其他股票）
        # 这里使用简单的排名方法
        relative_momentum = data.pct_change(periods=rel_lookback)

        # 初始化信号
        signals = pd.DataFrame(SignalType.HOLD.value, index=data.index, columns=data.columns)

        # 生成信号
        for i in range(max(abs_lookback, rel_lookback), len(data)):
            current_date = data.index[i]

            # 绝对动量筛选：过去abs_lookback期收益率为正
            abs_pos = absolute_momentum.iloc[i] > 0

            # 相对动量筛选：过去rel_lookback期收益率排名前top_n
            rel_momentum_current = relative_momentum.iloc[i]
            rel_ranked = rel_momentum_current.dropna().sort_values(ascending=False)
            rel_top = rel_ranked.head(top_n).index

            # 买入信号：同时满足绝对动量和相对动量条件
            buy_stocks = [s for s in data.columns if s in abs_pos.index and abs_pos[s] and s in rel_top]

            for stock in buy_stocks:
                signals.loc[current_date, stock] = SignalType.BUY.value

        return signals


class MeanReversionStrategy(BaseStrategy):
    """均值回归策略"""

    def __init__(
        self,
        lookback_period: int = 20,
        zscore_threshold: float = 2.0,
        **kwargs
    ):
        """
        初始化均值回归策略

        Args:
            lookback_period: 回顾期
            zscore_threshold: Z分数阈值
        """
        params = {
            "lookback_period": lookback_period,
            "zscore_threshold": zscore_threshold
        }
        params.update(kwargs)
        super().__init__(name="MeanReversion", params=params)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        生成均值回归策略信号

        Args:
            data: 包含价格数据的DataFrame

        Returns:
            信号序列
        """
        if 'Close' not in data.columns:
            raise ValueError("数据必须包含'Close'列")

        close = data['Close']
        lookback = self.params["lookback_period"]
        threshold = self.params["zscore_threshold"]

        if len(close) < lookback + 1:
            logger.warning("数据长度不足，无法计算Z分数")
            return pd.Series(SignalType.HOLD.value, index=close.index)

        # 计算滚动均值和标准差
        rolling_mean = close.rolling(window=lookback).mean()
        rolling_std = close.rolling(window=lookback).std()

        # 计算Z分数
        zscore = (close - rolling_mean) / rolling_std

        # 生成信号
        signals = pd.Series(SignalType.HOLD.value, index=close.index)

        # Z分数低于阈值（超卖）时买入
        buy_signals = zscore < -threshold
        signals[buy_signals] = SignalType.BUY.value

        # Z分数高于阈值（超买）时卖出
        sell_signals = zscore > threshold
        signals[sell_signals] = SignalType.SELL.value

        # 确保有足够的计算周期
        signals.iloc[:lookback] = SignalType.HOLD.value

        return signals


class MultiFactorStrategy(BaseStrategy):
    """多因子策略"""

    def __init__(
        self,
        factors: List[str],
        weights: List[float],
        top_n: int = 10,
        rebalance_freq: str = 'M',  # M: 月, W: 周, D: 日
        **kwargs
    ):
        """
        初始化多因子策略

        Args:
            factors: 因子列表
            weights: 因子权重
            top_n: 选择前N只股票
            rebalance_freq: 再平衡频率
        """
        if len(factors) != len(weights):
            raise ValueError("因子列表和权重列表长度必须相同")
        if abs(sum(weights) - 1.0) > 0.001:
            raise ValueError("权重之和必须为1")

        params = {
            "factors": factors,
            "weights": weights,
            "top_n": top_n,
            "rebalance_freq": rebalance_freq
        }
        params.update(kwargs)
        super().__init__(name="MultiFactor", params=params)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        生成多因子策略信号

        Args:
            data: 包含因子数据的DataFrame，列为因子值

        Returns:
            信号序列
        """
        if data.empty:
            logger.warning("输入数据为空")
            return pd.Series(dtype=float)

        # 检查是否包含所有需要的因子
        required_factors = self.params["factors"]
        missing_factors = [f for f in required_factors if f not in data.columns]
        if missing_factors:
            raise ValueError(f"缺少因子: {missing_factors}")

        data = data.sort_index()
        top_n = self.params["top_n"]
        weights = self.params["weights"]

        # 初始化信号
        signals = pd.Series(SignalType.HOLD.value, index=data.index)

        # 根据再平衡频率生成信号
        rebalance_freq = self.params["rebalance_freq"]
        if rebalance_freq == 'D':
            # 每日再平衡
            rebalance_dates = data.index
        elif rebalance_freq == 'W':
            # 每周再平衡
            rebalance_dates = data.index[data.index.weekday == 0]  # 周一
        elif rebalance_freq == 'M':
            # 每月再平衡
            rebalance_dates = data.index[data.index.is_month_start]
        else:
            raise ValueError(f"不支持的再平衡频率: {rebalance_freq}")

        for date in rebalance_dates:
            if date not in data.index:
                continue

            # 获取当前日期的因子数据
            factor_data = data.loc[date, required_factors]

            # 计算综合得分
            scores = pd.Series(0.0, index=factor_data.index)

            for factor, weight in zip(required_factors, weights):
                factor_values = factor_data[factor]

                # 标准化因子值（Z分数）
                if factor_values.std() > 0:
                    normalized = (factor_values - factor_values.mean()) / factor_values.std()
                else:
                    normalized = pd.Series(0.0, index=factor_values.index)

                # 累加加权得分
                scores += normalized * weight

            # 选择得分最高的top_n只股票
            top_stocks = scores.nlargest(top_n).index

            # 标记为买入信号
            signals.loc[date] = SignalType.BUY.value

        return signals


# 测试函数
def test_momentum_strategies():
    """测试动量策略"""
    import logging
    logging.basicConfig(level=logging.INFO)

    # 创建示例数据
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)

    # 创建多只股票数据
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    data = pd.DataFrame(
        np.random.randn(100, len(stocks)).cumsum(axis=0) + 100,
        index=dates,
        columns=stocks
    )

    print("测试动量策略...")
    strategy = MomentumStrategy(lookback_period=10, holding_period=5, top_n=2)
    signals = strategy.generate_signals(data)
    print(f"信号数据形状: {signals.shape}")
    print(f"买入信号总数: {(signals == SignalType.BUY.value).sum().sum()}")

    print("\n测试双动量策略...")
    dual_strategy = DualMomentumStrategy(absolute_lookback=10, relative_lookback=30, top_n=2)
    dual_signals = dual_strategy.generate_signals(data)
    print(f"双动量信号形状: {dual_signals.shape}")
    print(f"双动量买入信号总数: {(dual_signals == SignalType.BUY.value).sum().sum()}")

    print("\n测试均值回归策略（单只股票）...")
    single_data = pd.DataFrame({'Close': data['AAPL']})
    mean_rev_strategy = MeanReversionStrategy(lookback_period=20, zscore_threshold=1.5)
    mean_rev_signals = mean_rev_strategy.generate_signals(single_data)
    print(f"均值回归信号形状: {mean_rev_signals.shape}")
    print(f"买入信号数量: {(mean_rev_signals == SignalType.BUY.value).sum()}")
    print(f"卖出信号数量: {(mean_rev_signals == SignalType.SELL.value).sum()}")

    print("\n测试多因子策略...")
    # 创建因子数据
    factor_data = pd.DataFrame({
        'factor1': np.random.randn(100),
        'factor2': np.random.randn(100),
        'factor3': np.random.randn(100)
    }, index=dates)

    multi_factor_strategy = MultiFactorStrategy(
        factors=['factor1', 'factor2', 'factor3'],
        weights=[0.4, 0.3, 0.3],
        top_n=1,
        rebalance_freq='M'
    )
    multi_signals = multi_factor_strategy.generate_signals(factor_data)
    print(f"多因子信号形状: {multi_signals.shape}")
    print(f"买入信号数量: {(multi_signals == SignalType.BUY.value).sum()}")

    # 测试策略执行
    print("\n测试策略执行...")
    capital = 1000000
    commission_rate = 0.0003

    for i in range(min(30, len(signals))):
        date = signals.index[i]
        for stock in stocks:
            signal_value = signals.loc[date, stock]
            price = data.loc[date, stock]

            if signal_value == SignalType.BUY.value:
                trade = strategy.execute_trade(
                    symbol=stock,
                    date=date,
                    signal=SignalType.BUY,
                    price=price,
                    capital=capital,
                    commission_rate=commission_rate
                )
                if trade:
                    capital -= trade.value + trade.commission

    print(f"最终资金: {capital:.2f}")
    print(f"总交易次数: {len(strategy.trade_history)}")

    # 测试性能指标
    print("\n测试性能指标计算...")
    equity_curve = pd.Series(np.linspace(1000000, 1200000, 100), index=dates[:100])
    metrics = strategy.calculate_performance_metrics(equity_curve)
    print("动量策略性能指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    test_momentum_strategies()