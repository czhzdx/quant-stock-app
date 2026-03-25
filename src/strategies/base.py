"""策略基类模块"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """信号类型枚举"""
    BUY = 1      # 买入信号
    SELL = -1    # 卖出信号
    HOLD = 0     # 持有信号
    EXIT = -2    # 退出信号


@dataclass
class Position:
    """持仓信息"""
    symbol: str           # 股票代码
    entry_price: float    # 入场价格
    entry_date: pd.Timestamp  # 入场日期
    quantity: int         # 持仓数量
    current_price: float  # 当前价格
    pnl: float           # 盈亏
    pnl_pct: float       # 盈亏百分比


@dataclass
class Trade:
    """交易记录"""
    symbol: str           # 股票代码
    trade_date: pd.Timestamp  # 交易日期
    trade_type: SignalType    # 交易类型
    price: float         # 交易价格
    quantity: int        # 交易数量
    commission: float    # 手续费
    value: float        # 交易金额


class BaseStrategy(ABC):
    """策略基类"""

    def __init__(self, name: str = None, params: Optional[Dict] = None):
        """
        初始化策略

        Args:
            name: 策略名称
            params: 策略参数
        """
        self.name = name or self.__class__.__name__
        self.params = params or {}
        self.positions = {}  # 当前持仓
        self.trade_history = []  # 交易历史
        self.signals = None  # 信号序列
        self.equity_curve = None  # 权益曲线
        self.performance_metrics = {}  # 性能指标

        # 初始化日志
        self.logger = logging.getLogger(f"strategy.{self.name}")

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        生成交易信号

        Args:
            data: 包含因子数据的DataFrame

        Returns:
            信号序列 (1: 买入, -1: 卖出, 0: 持有)
        """
        pass

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        准备数据（可重写）

        Args:
            data: 原始数据

        Returns:
            处理后的数据
        """
        # 默认实现：复制数据并确保索引是日期时间
        processed_data = data.copy()

        if not isinstance(processed_data.index, pd.DatetimeIndex):
            if 'Date' in processed_data.columns:
                processed_data.index = pd.to_datetime(processed_data['Date'])
            elif 'date' in processed_data.columns:
                processed_data.index = pd.to_datetime(processed_data['date'])

        # 按日期排序
        processed_data = processed_data.sort_index()

        # 填充缺失值
        processed_data = processed_data.ffill().bfill()

        return processed_data

    def calculate_position_size(
        self,
        capital: float,
        price: float,
        risk_per_trade: float = 0.02
    ) -> int:
        """
        计算仓位大小

        Args:
            capital: 可用资金
            price: 股票价格
            risk_per_trade: 每笔交易风险比例

        Returns:
            持仓数量
        """
        if price <= 0:
            return 0

        # 基于风险管理的仓位计算
        position_value = capital * risk_per_trade
        quantity = int(position_value / price)

        # 确保最小交易单位（A股100股，美股1股）
        min_lot = 100 if price < 1 else 1  # 简化处理
        quantity = max(min_lot, quantity // min_lot * min_lot)

        return quantity

    def execute_trade(
        self,
        symbol: str,
        date: pd.Timestamp,
        signal: SignalType,
        price: float,
        capital: float,
        commission_rate: float = 0.0003
    ) -> Optional[Trade]:
        """
        执行交易

        Args:
            symbol: 股票代码
            date: 交易日期
            signal: 交易信号
            price: 交易价格
            capital: 可用资金
            commission_rate: 手续费率

        Returns:
            交易记录或None
        """
        if price <= 0 or capital <= 0:
            self.logger.warning(f"无效的价格或资金: price={price}, capital={capital}")
            return None

        # 计算仓位
        quantity = self.calculate_position_size(capital, price)

        if quantity <= 0:
            return None

        # 计算交易金额和手续费
        trade_value = quantity * price
        commission = trade_value * commission_rate

        # 创建交易记录
        trade = Trade(
            symbol=symbol,
            trade_date=date,
            trade_type=signal,
            price=price,
            quantity=quantity,
            commission=commission,
            value=trade_value
        )

        # 更新持仓
        if signal == SignalType.BUY:
            if symbol in self.positions:
                # 加仓
                position = self.positions[symbol]
                avg_price = (position.entry_price * position.quantity + price * quantity) / \
                           (position.quantity + quantity)
                position.quantity += quantity
                position.entry_price = avg_price
            else:
                # 新建持仓
                self.positions[symbol] = Position(
                    symbol=symbol,
                    entry_price=price,
                    entry_date=date,
                    quantity=quantity,
                    current_price=price,
                    pnl=0,
                    pnl_pct=0
                )

        elif signal == SignalType.SELL or signal == SignalType.EXIT:
            if symbol in self.positions:
                position = self.positions[symbol]

                # 计算盈亏
                pnl = (price - position.entry_price) * min(quantity, position.quantity)
                pnl_pct = (price / position.entry_price - 1) * 100

                # 更新持仓
                if quantity >= position.quantity:
                    # 平仓
                    del self.positions[symbol]
                else:
                    # 部分卖出
                    position.quantity -= quantity

                # 记录盈亏
                trade.pnl = pnl
                trade.pnl_pct = pnl_pct

        # 添加到交易历史
        self.trade_history.append(trade)

        self.logger.info(
            f"{date.date()} {symbol} {signal.name}: "
            f"价格={price:.2f}, 数量={quantity}, 金额={trade_value:.2f}, 手续费={commission:.2f}"
        )

        return trade

    def update_positions(self, date: pd.Timestamp, prices: Dict[str, float]):
        """
        更新持仓市值和盈亏

        Args:
            date: 当前日期
            prices: 股票价格字典 {symbol: price}
        """
        total_pnl = 0
        total_value = 0

        for symbol, position in list(self.positions.items()):
            if symbol in prices:
                current_price = prices[symbol]
                position.current_price = current_price
                position.pnl = (current_price - position.entry_price) * position.quantity
                position.pnl_pct = (current_price / position.entry_price - 1) * 100

                total_pnl += position.pnl
                total_value += current_price * position.quantity
            else:
                self.logger.warning(f"无法获取 {symbol} 的价格，跳过更新")

        return total_pnl, total_value

    def calculate_performance_metrics(
        self,
        equity_curve: pd.Series,
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        计算性能指标

        Args:
            equity_curve: 权益曲线
            benchmark_returns: 基准收益率序列

        Returns:
            性能指标字典
        """
        if equity_curve.empty or len(equity_curve) < 2:
            return {}

        # 计算收益率
        returns = equity_curve.pct_change().dropna()

        # 基本指标
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
        annual_return = ((1 + total_return/100) ** (252/len(returns)) - 1) * 100

        # 波动率
        volatility = returns.std() * np.sqrt(252) * 100

        # 夏普比率（假设无风险利率为0）
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        # 最大回撤
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative / running_max - 1) * 100
        max_drawdown = drawdown.min()

        # 胜率
        if self.trade_history:
            winning_trades = [t for t in self.trade_history if t.pnl > 0]
            win_rate = len(winning_trades) / len(self.trade_history) * 100
        else:
            win_rate = 0

        # 盈亏比
        if winning_trades:
            avg_win = np.mean([t.pnl for t in winning_trades])
            losing_trades = [t for t in self.trade_history if t.pnl <= 0]
            avg_loss = abs(np.mean([t.pnl for t in losing_trades])) if losing_trades else 0
            profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
        else:
            profit_loss_ratio = 0

        # Alpha/Beta（如果有基准）
        alpha = beta = 0
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            try:
                # 对齐索引
                aligned_returns = returns.reindex(benchmark_returns.index).dropna()
                aligned_benchmark = benchmark_returns.reindex(aligned_returns.index)

                if len(aligned_returns) > 1:
                    cov_matrix = np.cov(aligned_returns, aligned_benchmark)
                    beta = cov_matrix[0, 1] / cov_matrix[1, 1]
                    alpha = (aligned_returns.mean() - beta * aligned_benchmark.mean()) * 252 * 100
            except:
                pass

        metrics = {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_loss_ratio": profit_loss_ratio,
            "alpha": alpha,
            "beta": beta,
            "total_trades": len(self.trade_history),
            "winning_trades": len(winning_trades) if self.trade_history else 0,
            "losing_trades": len(self.trade_history) - len(winning_trades) if self.trade_history else 0,
        }

        self.performance_metrics = metrics
        return metrics

    def get_position_summary(self) -> pd.DataFrame:
        """获取持仓摘要"""
        positions_list = []
        for symbol, position in self.positions.items():
            positions_list.append({
                "symbol": position.symbol,
                "entry_price": position.entry_price,
                "entry_date": position.entry_date,
                "quantity": position.quantity,
                "current_price": position.current_price,
                "market_value": position.current_price * position.quantity,
                "pnl": position.pnl,
                "pnl_pct": position.pnl_pct
            })

        return pd.DataFrame(positions_list)

    def get_trade_summary(self) -> pd.DataFrame:
        """获取交易摘要"""
        trades_list = []
        for trade in self.trade_history:
            trades_list.append({
                "symbol": trade.symbol,
                "date": trade.trade_date,
                "type": trade.trade_type.name,
                "price": trade.price,
                "quantity": trade.quantity,
                "value": trade.value,
                "commission": trade.commission,
                "pnl": getattr(trade, 'pnl', 0),
                "pnl_pct": getattr(trade, 'pnl_pct', 0)
            })

        return pd.DataFrame(trades_list)

    def reset(self):
        """重置策略状态"""
        self.positions = {}
        self.trade_history = []
        self.signals = None
        self.equity_curve = None
        self.performance_metrics = {}
        self.logger.info("策略已重置")

    def __str__(self) -> str:
        """字符串表示"""
        return f"{self.name}(params={self.params})"

    def __repr__(self) -> str:
        """表示"""
        return self.__str__()


# 示例策略
class ExampleStrategy(BaseStrategy):
    """示例策略：简单移动平均线交叉策略"""

    def __init__(self, fast_period: int = 10, slow_period: int = 30, **kwargs):
        """
        初始化移动平均线交叉策略

        Args:
            fast_period: 快速均线周期
            slow_period: 慢速均线周期
        """
        params = {
            "fast_period": fast_period,
            "slow_period": slow_period
        }
        params.update(kwargs)
        super().__init__(name="MA_Crossover", params=params)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        生成移动平均线交叉信号

        Args:
            data: 包含价格数据的DataFrame

        Returns:
            信号序列
        """
        if 'Close' not in data.columns:
            raise ValueError("数据必须包含'Close'列")

        close = data['Close']
        fast_period = self.params.get("fast_period", 10)
        slow_period = self.params.get("slow_period", 30)

        # 计算移动平均线
        fast_ma = close.rolling(window=fast_period).mean()
        slow_ma = close.rolling(window=slow_period).mean()

        # 生成信号
        signals = pd.Series(SignalType.HOLD.value, index=close.index)

        # 金叉：快速均线上穿慢速均线，买入信号
        golden_cross = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        signals[golden_cross] = SignalType.BUY.value

        # 死叉：快速均线下穿慢速均线，卖出信号
        death_cross = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
        signals[death_cross] = SignalType.SELL.value

        # 确保有足够的计算周期
        signals.iloc[:max(fast_period, slow_period)] = SignalType.HOLD.value

        return signals


# 测试函数
def test_base_strategy():
    """测试策略基类"""
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

    # 测试示例策略
    print("测试移动平均线交叉策略...")
    strategy = ExampleStrategy(fast_period=5, slow_period=20)

    # 准备数据
    processed_data = strategy.prepare_data(data)
    print(f"处理后的数据形状: {processed_data.shape}")

    # 生成信号
    signals = strategy.generate_signals(processed_data)
    print(f"信号序列形状: {signals.shape}")
    print(f"买入信号数量: {(signals == SignalType.BUY.value).sum()}")
    print(f"卖出信号数量: {(signals == SignalType.SELL.value).sum()}")

    # 测试交易执行
    print("\n测试交易执行...")
    capital = 1000000
    commission_rate = 0.0003

    for i in range(min(20, len(signals))):
        date = processed_data.index[i]
        signal_value = signals.iloc[i]
        price = processed_data['Close'].iloc[i]

        if signal_value == SignalType.BUY.value:
            trade = strategy.execute_trade(
                symbol="AAPL",
                date=date,
                signal=SignalType.BUY,
                price=price,
                capital=capital,
                commission_rate=commission_rate
            )
            if trade:
                capital -= trade.value + trade.commission
                print(f"买入交易: {trade.trade_date.date()} 价格={trade.price:.2f} 数量={trade.quantity}")

        elif signal_value == SignalType.SELL.value and "AAPL" in strategy.positions:
            trade = strategy.execute_trade(
                symbol="AAPL",
                date=date,
                signal=SignalType.SELL,
                price=price,
                capital=capital,
                commission_rate=commission_rate
            )
            if trade:
                capital += trade.value - trade.commission
                print(f"卖出交易: {trade.trade_date.date()} 价格={trade.price:.2f} 数量={trade.quantity} 盈亏={trade.pnl:.2f}")

    # 测试持仓更新
    print("\n测试持仓更新...")
    prices = {"AAPL": 105.0}
    total_pnl, total_value = strategy.update_positions(dates[19], prices)
    print(f"总盈亏: {total_pnl:.2f}, 总市值: {total_value:.2f}")

    # 测试获取持仓摘要
    print("\n测试获取持仓摘要...")
    position_summary = strategy.get_position_summary()
    if not position_summary.empty:
        print(f"持仓摘要:\n{position_summary}")

    # 测试获取交易摘要
    print("\n测试获取交易摘要...")
    trade_summary = strategy.get_trade_summary()
    if not trade_summary.empty:
        print(f"交易摘要:\n{trade_summary}")

    # 测试性能指标计算
    print("\n测试性能指标计算...")
    # 创建示例权益曲线
    equity_curve = pd.Series(
        np.linspace(1000000, 1200000, 100),
        index=dates
    )
    benchmark_returns = pd.Series(np.random.randn(100) * 0.001, index=dates)

    metrics = strategy.calculate_performance_metrics(equity_curve, benchmark_returns)
    print("性能指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # 测试重置策略
    print("\n测试重置策略...")
    strategy.reset()
    print(f"重置后持仓数量: {len(strategy.positions)}")
    print(f"重置后交易历史数量: {len(strategy.trade_history)}")


if __name__ == "__main__":
    test_base_strategy()