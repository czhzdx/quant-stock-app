"""回测引擎模块"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from src.strategies.base import BaseStrategy, SignalType
from src.utils.config_loader import get_config

logger = logging.getLogger(__name__)


class BacktestMode(Enum):
    """回测模式"""
    SINGLE_STOCK = "single_stock"      # 单只股票回测
    MULTI_STOCK = "multi_stock"        # 多只股票回测
    PORTFOLIO = "portfolio"            # 投资组合回测


@dataclass
class BacktestResult:
    """回测结果"""
    equity_curve: pd.Series                    # 权益曲线
    returns: pd.Series                         # 收益率序列
    positions: pd.DataFrame                    # 持仓记录
    trades: pd.DataFrame                       # 交易记录
    performance_metrics: Dict[str, float]      # 性能指标
    benchmark_returns: Optional[pd.Series] = None  # 基准收益率


class Backtester:
    """回测引擎"""

    def __init__(
        self,
        strategy: BaseStrategy,
        initial_capital: float = 1000000,
        commission_rate: float = 0.0003,
        slippage_rate: float = 0.0001,
        benchmark_symbol: Optional[str] = None
    ):
        """
        初始化回测引擎

        Args:
            strategy: 交易策略
            initial_capital: 初始资金
            commission_rate: 手续费率
            slippage_rate: 滑点率
            benchmark_symbol: 基准指数代码
        """
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.benchmark_symbol = benchmark_symbol

        # 回测状态
        self.capital = initial_capital
        self.positions = {}  # 当前持仓 {symbol: quantity}
        self.position_values = {}  # 持仓市值 {symbol: value}
        self.equity_curve = []  # 权益曲线
        self.trade_records = []  # 交易记录
        self.position_records = []  # 持仓记录
        self.dates = []  # 日期序列

        # 加载配置
        self.config = get_config()  # 获取整个配置字典
        self.risk_config = self.config.get("backtest", {}).get("risk_management", {})

        logger.info(f"初始化回测引擎: 策略={strategy.name}, 初始资金={initial_capital}")

    def run(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        mode: BacktestMode = BacktestMode.SINGLE_STOCK
    ) -> BacktestResult:
        """
        运行回测

        Args:
            data: 股票数据
            start_date: 开始日期
            end_date: 结束日期
            mode: 回测模式

        Returns:
            回测结果
        """
        logger.info(f"开始回测: 模式={mode.value}")

        # 重置状态
        self._reset()

        # 准备数据
        prepared_data = self._prepare_data(data, mode)
        if prepared_data is None:
            raise ValueError("数据准备失败")

        # 过滤日期范围
        if start_date:
            start_date = pd.to_datetime(start_date)
            prepared_data = prepared_data[prepared_data.index >= start_date]
        if end_date:
            end_date = pd.to_datetime(end_date)
            prepared_data = prepared_data[prepared_data.index <= end_date]

        if prepared_data.empty:
            raise ValueError("过滤后的数据为空")

        logger.info(f"回测日期范围: {prepared_data.index[0]} 到 {prepared_data.index[-1]}")
        logger.info(f"数据条数: {len(prepared_data)}")

        # 运行回测
        if mode == BacktestMode.SINGLE_STOCK:
            result = self._run_single_stock(prepared_data)
        elif mode == BacktestMode.MULTI_STOCK:
            result = self._run_multi_stock(prepared_data)
        elif mode == BacktestMode.PORTFOLIO:
            result = self._run_portfolio(prepared_data)
        else:
            raise ValueError(f"不支持的回测模式: {mode}")

        # 计算性能指标
        result.performance_metrics = self._calculate_performance_metrics(result)

        logger.info("回测完成")
        return result

    def _prepare_data(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        mode: BacktestMode
    ) -> Optional[pd.DataFrame]:
        """准备回测数据"""
        if mode == BacktestMode.SINGLE_STOCK:
            if isinstance(data, pd.DataFrame):
                # 单只股票数据
                if 'Close' not in data.columns:
                    raise ValueError("单只股票数据必须包含'Close'列")
                return data
            else:
                raise ValueError("单只股票回测需要DataFrame格式的数据")

        elif mode in [BacktestMode.MULTI_STOCK, BacktestMode.PORTFOLIO]:
            if isinstance(data, dict):
                # 多只股票数据字典
                # 对齐所有股票的日期
                common_dates = None
                for symbol, df in data.items():
                    if common_dates is None:
                        common_dates = set(df.index)
                    else:
                        common_dates = common_dates.intersection(set(df.index))

                if not common_dates:
                    raise ValueError("股票数据没有共同的日期")

                common_dates = sorted(common_dates)
                prepared_data = pd.DataFrame(index=common_dates)

                for symbol, df in data.items():
                    if 'Close' not in df.columns:
                        raise ValueError(f"股票{symbol}数据必须包含'Close'列")

                    # 对齐到共同日期
                    aligned_df = df.loc[common_dates]
                    prepared_data[symbol] = aligned_df['Close']

                return prepared_data
            elif isinstance(data, pd.DataFrame):
                # 已经是多股票格式
                return data
            else:
                raise ValueError("多股票回测需要字典或DataFrame格式的数据")

        return None

    def _run_single_stock(self, data: pd.DataFrame) -> BacktestResult:
        """运行单只股票回测"""
        logger.info("运行单只股票回测")

        # 生成信号
        signals = self.strategy.generate_signals(data)

        for date in data.index:
            self.dates.append(date)
            current_price = data.loc[date, 'Close']

            # 更新持仓市值
            self._update_positions(date, {'Close': current_price})

            # 获取当前信号
            if date in signals.index:
                # 对于单只股票回测，信号应该是标量值
                if isinstance(signals, pd.Series):
                    signal = signals.loc[date]
                else:
                    # 如果是DataFrame，取第一列（假设只有一列）
                    signal = signals.iloc[signals.index.get_loc(date), 0]
            else:
                signal = SignalType.HOLD.value

            # 执行交易
            if pd.notna(signal) and signal == SignalType.BUY.value:
                self._execute_trade(date, 'Close', current_price, SignalType.BUY)
            elif pd.notna(signal) and signal == SignalType.SELL.value:
                self._execute_trade(date, 'Close', current_price, SignalType.SELL)

            # 记录权益
            self._record_equity(date)

        return self._collect_results(data)

    def _run_multi_stock(self, data: pd.DataFrame) -> BacktestResult:
        """运行多只股票回测"""
        logger.info(f"运行多只股票回测，股票数量: {len(data.columns)}")

        # 生成信号
        signals = self.strategy.generate_signals(data)

        for date in data.index:
            self.dates.append(date)
            current_prices = data.loc[date].to_dict()

            # 更新持仓市值
            self._update_positions(date, current_prices)

            # 获取当前信号
            if date in signals.index:
                date_signals = signals.loc[date]
            else:
                date_signals = pd.Series(SignalType.HOLD.value, index=data.columns)

            # 执行交易
            for symbol, price in current_prices.items():
                signal = date_signals.get(symbol, SignalType.HOLD.value)

                if signal == SignalType.BUY.value:
                    self._execute_trade(date, symbol, price, SignalType.BUY)
                elif signal == SignalType.SELL.value:
                    self._execute_trade(date, symbol, price, SignalType.SELL)

            # 记录权益
            self._record_equity(date)

        return self._collect_results(data)

    def _run_portfolio(self, data: pd.DataFrame) -> BacktestResult:
        """运行投资组合回测"""
        logger.info(f"运行投资组合回测，股票数量: {len(data.columns)}")

        # 生成信号
        signals = self.strategy.generate_signals(data)

        for date in data.index:
            self.dates.append(date)
            current_prices = data.loc[date].to_dict()

            # 更新持仓市值
            self._update_positions(date, current_prices)

            # 获取当前信号
            if date in signals.index:
                signal = signals.loc[date]
            else:
                signal = SignalType.HOLD.value

            # 执行交易（投资组合级别）
            if signal == SignalType.BUY.value:
                # 投资组合买入信号，平均分配到所有股票
                for symbol, price in current_prices.items():
                    self._execute_trade(date, symbol, price, SignalType.BUY)
            elif signal == SignalType.SELL.value:
                # 投资组合卖出信号，卖出所有持仓
                for symbol, price in current_prices.items():
                    if symbol in self.positions and self.positions[symbol] > 0:
                        self._execute_trade(date, symbol, price, SignalType.SELL)

            # 记录权益
            self._record_equity(date)

        return self._collect_results(data)

    def _execute_trade(
        self,
        date: pd.Timestamp,
        symbol: str,
        price: float,
        signal_type: SignalType
    ):
        """执行交易"""
        if price <= 0:
            return

        # 应用滑点
        if signal_type == SignalType.BUY:
            execution_price = price * (1 + self.slippage_rate)
        else:
            execution_price = price * (1 - self.slippage_rate)

        # 计算交易数量
        if signal_type == SignalType.BUY:
            # 买入：基于可用资金计算数量
            position_value = self.capital * self.risk_config.get("max_position_size", 0.1)
            quantity = int(position_value / execution_price)
            if quantity <= 0:
                return
        else:
            # 卖出：卖出全部持仓
            quantity = self.positions.get(symbol, 0)
            if quantity <= 0:
                return

        # 计算交易金额和手续费
        trade_value = quantity * execution_price
        commission = trade_value * self.commission_rate

        # 更新资金和持仓
        if signal_type == SignalType.BUY:
            if trade_value + commission > self.capital:
                logger.warning(f"资金不足，无法买入 {symbol}")
                return

            self.capital -= (trade_value + commission)
            self.positions[symbol] = self.positions.get(symbol, 0) + quantity
        else:  # SELL
            self.capital += (trade_value - commission)
            self.positions[symbol] = self.positions.get(symbol, 0) - quantity

            # 清理零持仓
            if self.positions[symbol] == 0:
                del self.positions[symbol]

        # 记录交易
        trade_record = {
            'date': date,
            'symbol': symbol,
            'type': signal_type.name,
            'price': execution_price,
            'quantity': quantity,
            'value': trade_value,
            'commission': commission,
            'capital': self.capital
        }
        self.trade_records.append(trade_record)

        logger.debug(
            f"{date.date()} {signal_type.name} {symbol}: "
            f"价格={execution_price:.2f}, 数量={quantity}, "
            f"金额={trade_value:.2f}, 手续费={commission:.2f}, "
            f"剩余资金={self.capital:.2f}"
        )

    def _update_positions(self, date: pd.Timestamp, prices: Dict[str, float]):
        """更新持仓市值"""
        total_position_value = 0

        for symbol, quantity in list(self.positions.items()):
            if symbol in prices:
                price = prices[symbol]
                position_value = quantity * price
                total_position_value += position_value

                # 记录持仓
                position_record = {
                    'date': date,
                    'symbol': symbol,
                    'quantity': quantity,
                    'price': price,
                    'value': position_value
                }
                self.position_records.append(position_record)
            else:
                logger.warning(f"无法获取 {symbol} 的价格，跳过更新")

        self.position_values[date] = total_position_value

    def _record_equity(self, date: pd.Timestamp):
        """记录权益"""
        total_equity = self.capital + self.position_values.get(date, 0)
        self.equity_curve.append({
            'date': date,
            'equity': total_equity,
            'capital': self.capital,
            'position_value': self.position_values.get(date, 0)
        })

    def _collect_results(self, data: pd.DataFrame) -> BacktestResult:
        """收集回测结果"""
        # 权益曲线
        equity_df = pd.DataFrame(self.equity_curve).set_index('date')
        equity_curve = equity_df['equity']

        # 收益率序列
        returns = equity_curve.pct_change().fillna(0)

        # 持仓记录
        positions_df = pd.DataFrame(self.position_records)
        if not positions_df.empty:
            positions_df = positions_df.set_index('date')

        # 交易记录
        trades_df = pd.DataFrame(self.trade_records)
        if not trades_df.empty:
            trades_df = trades_df.set_index('date')

        # 基准收益率（如果提供了基准代码）
        benchmark_returns = None
        if self.benchmark_symbol:
            try:
                # 这里可以添加获取基准数据的逻辑
                pass
            except Exception as e:
                logger.warning(f"获取基准数据失败: {e}")

        return BacktestResult(
            equity_curve=equity_curve,
            returns=returns,
            positions=positions_df,
            trades=trades_df,
            performance_metrics={},
            benchmark_returns=benchmark_returns
        )

    def _calculate_performance_metrics(self, result: BacktestResult) -> Dict[str, float]:
        """计算性能指标"""
        if result.equity_curve.empty or len(result.equity_curve) < 2:
            return {}

        equity = result.equity_curve
        returns = result.returns

        # 基本指标
        total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
        annual_return = self._calculate_annual_return(equity)

        # 波动率
        volatility = returns.std() * np.sqrt(252) * 100

        # 夏普比率
        sharpe_ratio = self._calculate_sharpe_ratio(returns)

        # 最大回撤
        max_drawdown = self._calculate_max_drawdown(equity)

        # 索提诺比率
        sortino_ratio = self._calculate_sortino_ratio(returns)

        # 卡尔玛比率
        calmar_ratio = self._calculate_calmar_ratio(annual_return, max_drawdown)

        # 胜率
        win_rate, profit_loss_ratio = self._calculate_win_rate(result.trades)

        # Alpha/Beta
        alpha, beta = 0, 0
        if result.benchmark_returns is not None:
            alpha, beta = self._calculate_alpha_beta(returns, result.benchmark_returns)

        metrics = {
            '初始资金': self.initial_capital,
            '最终权益': equity.iloc[-1],
            '总收益率%': total_return,
            '年化收益率%': annual_return,
            '年化波动率%': volatility,
            '夏普比率': sharpe_ratio,
            '最大回撤%': max_drawdown,
            '索提诺比率': sortino_ratio,
            '卡尔玛比率': calmar_ratio,
            '胜率%': win_rate,
            '盈亏比': profit_loss_ratio,
            '总交易次数': len(result.trades) if not result.trades.empty else 0,
            'Alpha': alpha,
            'Beta': beta,
            '交易天数': len(equity),
        }

        return metrics

    def _calculate_annual_return(self, equity: pd.Series) -> float:
        """计算年化收益率"""
        total_return = equity.iloc[-1] / equity.iloc[0] - 1
        days = (equity.index[-1] - equity.index[0]).days
        if days > 0:
            annual_return = (1 + total_return) ** (365 / days) - 1
            return annual_return * 100
        return 0

    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        excess_returns = returns - risk_free_rate / 252
        if returns.std() > 0:
            sharpe = excess_returns.mean() / returns.std() * np.sqrt(252)
            return sharpe
        return 0

    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        """计算最大回撤"""
        cumulative = (1 + equity.pct_change().fillna(0)).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative / running_max - 1) * 100
        return drawdown.min()

    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """计算索提诺比率"""
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino = excess_returns.mean() / downside_returns.std() * np.sqrt(252)
            return sortino
        return 0

    def _calculate_calmar_ratio(self, annual_return: float, max_drawdown: float) -> float:
        """计算卡尔玛比率"""
        if max_drawdown < 0:
            return annual_return / abs(max_drawdown)
        return 0

    def _calculate_win_rate(self, trades: pd.DataFrame) -> Tuple[float, float]:
        """计算胜率和盈亏比"""
        if trades.empty or 'value' not in trades.columns:
            return 0, 0

        # 这里简化处理，实际需要根据交易盈亏计算
        return 0, 0

    def _calculate_alpha_beta(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Tuple[float, float]:
        """计算Alpha和Beta"""
        try:
            # 对齐数据
            aligned_returns = strategy_returns.reindex(benchmark_returns.index).dropna()
            aligned_benchmark = benchmark_returns.reindex(aligned_returns.index)

            if len(aligned_returns) > 1:
                cov_matrix = np.cov(aligned_returns, aligned_benchmark)
                beta = cov_matrix[0, 1] / cov_matrix[1, 1]
                alpha = (aligned_returns.mean() - beta * aligned_benchmark.mean()) * 252
                return alpha * 100, beta
        except:
            pass

        return 0, 0

    def _reset(self):
        """重置回测状态"""
        self.capital = self.initial_capital
        self.positions = {}
        self.position_values = {}
        self.equity_curve = []
        self.trade_records = []
        self.position_records = []
        self.dates = []
        self.strategy.reset()

    def get_summary(self, result: BacktestResult) -> pd.DataFrame:
        """获取回测摘要"""
        metrics = result.performance_metrics

        summary_data = []
        for key, value in metrics.items():
            if isinstance(value, float):
                if '率' in key or '比' in key:
                    summary_data.append([key, f"{value:.4f}"])
                elif '资金' in key or '权益' in key:
                    summary_data.append([key, f"{value:,.2f}"])
                else:
                    summary_data.append([key, f"{value:.2f}"])
            else:
                summary_data.append([key, str(value)])

        return pd.DataFrame(summary_data, columns=['指标', '值'])


# 测试函数
def test_backtester():
    """测试回测引擎"""
    import logging
    logging.basicConfig(level=logging.INFO)

    # 创建示例数据
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)

    # 单只股票数据
    single_data = pd.DataFrame({
        'Open': np.random.randn(100).cumsum() + 100,
        'High': np.random.randn(100).cumsum() + 105,
        'Low': np.random.randn(100).cumsum() + 95,
        'Close': np.random.randn(100).cumsum() + 100,
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)

    # 多只股票数据
    stocks = ['AAPL', 'MSFT', 'GOOGL']
    multi_data = {}
    for stock in stocks:
        multi_data[stock] = pd.DataFrame({
            'Open': np.random.randn(100).cumsum() + 100,
            'High': np.random.randn(100).cumsum() + 105,
            'Low': np.random.randn(100).cumsum() + 95,
            'Close': np.random.randn(100).cumsum() + 100,
            'Volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)

    print("测试单只股票回测...")
    from src.strategies.momentum import MomentumStrategy

    strategy = MomentumStrategy(lookback_period=10, holding_period=5, top_n=1)
    backtester = Backtester(strategy, initial_capital=1000000)

    result = backtester.run(
        data=single_data,
        start_date='2023-01-10',
        end_date='2023-03-31',
        mode=BacktestMode.SINGLE_STOCK
    )

    print(f"权益曲线长度: {len(result.equity_curve)}")
    print(f"收益率序列长度: {len(result.returns)}")
    print(f"交易记录数量: {len(result.trades) if not result.trades.empty else 0}")

    summary = backtester.get_summary(result)
    print("\n回测摘要:")
    print(summary.to_string(index=False))

    print("\n测试多只股票回测...")
    strategy2 = MomentumStrategy(lookback_period=10, holding_period=5, top_n=2)
    backtester2 = Backtester(strategy2, initial_capital=1000000)

    result2 = backtester2.run(
        data=multi_data,
        start_date='2023-01-10',
        end_date='2023-03-31',
        mode=BacktestMode.MULTI_STOCK
    )

    print(f"多股票回测权益曲线长度: {len(result2.equity_curve)}")
    print(f"多股票回测交易记录数量: {len(result2.trades) if not result2.trades.empty else 0}")

    summary2 = backtester2.get_summary(result2)
    print("\n多股票回测摘要:")
    print(summary2.to_string(index=False))

    # 测试性能指标
    print("\n性能指标详情:")
    for key, value in result2.performance_metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    test_backtester()