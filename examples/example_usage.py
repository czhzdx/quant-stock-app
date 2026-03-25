"""量化选股软件使用示例"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

from src.data.data_fetcher import DataFetcher
from src.factors.factor_calculator import FactorCalculator, FactorType
from src.strategies.momentum import MomentumStrategy, DualMomentumStrategy, MeanReversionStrategy, MultiFactorStrategy
from src.backtest.backtester import Backtester, BacktestMode
from src.visualization.plotter import Plotter
from src.utils.config_loader import get_config_loader


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )


def example_data_fetcher():
    """示例：数据获取"""
    print("=" * 80)
    print("示例1: 数据获取")
    print("=" * 80)

    # 初始化数据获取器
    fetcher = DataFetcher(cache_enabled=False)  # 禁用缓存，使用模拟数据

    # 创建模拟数据（避免API限制）
    print("\n1. 创建模拟股票数据 (AAPL):")
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)

    aapl_data = pd.DataFrame({
        'Open': np.random.randn(100).cumsum() + 150,
        'High': np.random.randn(100).cumsum() + 155,
        'Low': np.random.randn(100).cumsum() + 145,
        'Close': np.random.randn(100).cumsum() + 150,
        'Volume': np.random.randint(10000000, 50000000, 100),
        'Adj Close': np.random.randn(100).cumsum() + 150
    }, index=dates)

    print(f"  数据形状: {aapl_data.shape}")
    print(f"  数据列: {aapl_data.columns.tolist()}")
    print(f"  日期范围: {aapl_data.index[0]} 到 {aapl_data.index[-1]}")
    print(f"  前5行数据:\n{aapl_data.head()}")

    # 创建多只股票模拟数据
    print("\n2. 创建多只股票模拟数据 (AAPL, MSFT, GOOGL):")
    symbols = ["AAPL", "MSFT", "GOOGL"]
    multi_data = {}

    for i, symbol in enumerate(symbols):
        multi_data[symbol] = pd.DataFrame({
            'Open': np.random.randn(100).cumsum() + 150 + i*50,
            'High': np.random.randn(100).cumsum() + 155 + i*50,
            'Low': np.random.randn(100).cumsum() + 145 + i*50,
            'Close': np.random.randn(100).cumsum() + 150 + i*50,
            'Volume': np.random.randint(10000000, 50000000, 100),
            'Adj Close': np.random.randn(100).cumsum() + 150 + i*50
        }, index=dates)
        print(f"  {symbol}: {multi_data[symbol].shape}")

    # 创建指数模拟数据
    print("\n3. 创建指数模拟数据 (沪深300):")
    index_data = pd.DataFrame({
        'Open': np.random.randn(100).cumsum() + 3000,
        'High': np.random.randn(100).cumsum() + 3050,
        'Low': np.random.randn(100).cumsum() + 2950,
        'Close': np.random.randn(100).cumsum() + 3000,
        'Volume': np.random.randint(100000000, 500000000, 100),
        'Adj Close': np.random.randn(100).cumsum() + 3000
    }, index=dates)
    print(f"  指数数据形状: {index_data.shape}")

    # 模拟基本面数据
    print("\n4. 模拟基本面数据 (AAPL):")
    fundamental = {
        "symbol": "AAPL",
        "company_name": "Apple Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "market_cap": 2800000000000,
        "pe_ratio": 28.5,
        "pb_ratio": 35.2,
        "dividend_yield": 0.005,
        "roe": 0.45,
        "profit_margins": 0.25,
        "revenue_growth": 0.08,
        "earnings_growth": 0.12,
        "debt_to_equity": 1.5,
        "current_ratio": 1.2,
    }
    print(f"  公司名称: {fundamental.get('company_name', 'N/A')}")
    print(f"  市值: {fundamental.get('market_cap', 'N/A'):,.0f}")
    print(f"  市盈率: {fundamental.get('pe_ratio', 'N/A')}")
    print(f"  市净率: {fundamental.get('pb_ratio', 'N/A')}")

    return aapl_data, multi_data


def example_factor_calculator(stock_data):
    """示例：因子计算"""
    print("\n" + "=" * 80)
    print("示例2: 因子计算")
    print("=" * 80)

    # 初始化因子计算器
    calculator = FactorCalculator()

    # 列出所有因子
    print("\n1. 列出所有因子:")
    all_factors = calculator.list_factors()
    print(f"  总共 {len(all_factors)} 个因子")

    technical_factors = calculator.list_factors(FactorType.TECHNICAL)
    print(f"  技术指标因子 ({len(technical_factors)} 个): {technical_factors[:5]}...")

    # 计算单个因子
    print("\n2. 计算单个因子 (MA20):")
    ma_20 = calculator.calculate_factor(stock_data, "ma_20")
    print(f"  MA20前5个值:\n{ma_20.head()}")

    # 计算多个因子
    print("\n3. 计算多个因子 (MA10, RSI14, MACD):")
    factor_names = ["ma_10", "rsi_14", "macd"]
    factors_df = calculator.calculate_multiple_factors(stock_data, factor_names)
    print(f"  因子数据形状: {factors_df.shape}")
    print(f"  因子数据前5行:\n{factors_df.head()}")

    # 计算所有因子
    print("\n4. 计算所有因子:")
    all_factors_df = calculator.calculate_all_factors(stock_data)
    print(f"  所有因子数据形状: {all_factors_df.shape}")
    print(f"  因子列: {all_factors_df.columns.tolist()}")

    # 注册自定义因子
    print("\n5. 注册自定义因子:")

    def custom_volatility(data: pd.DataFrame, period: int = 20) -> pd.Series:
        """自定义波动率因子"""
        returns = data['Close'].pct_change()
        volatility = returns.rolling(window=period).std() * np.sqrt(252)
        return volatility

    calculator.register_custom_factor(
        name="annual_volatility_20",
        description="20日年化波动率",
        calculation_func=lambda data: custom_volatility(data, 20),
        parameters={"period": 20}
    )

    # 计算自定义因子
    custom_factor = calculator.calculate_factor(stock_data, "annual_volatility_20")
    print(f"  自定义因子前5个值:\n{custom_factor.head()}")

    return factors_df


def example_strategies(stock_data, factor_data):
    """示例：策略"""
    print("\n" + "=" * 80)
    print("示例3: 策略")
    print("=" * 80)

    # 1. 动量策略
    print("\n1. 动量策略:")
    momentum_strategy = MomentumStrategy(
        lookback_period=20,
        holding_period=10,
        top_n=3
    )
    print(f"  策略名称: {momentum_strategy.name}")
    print(f"  策略参数: {momentum_strategy.params}")

    # 准备数据
    processed_data = momentum_strategy.prepare_data(stock_data)
    print(f"  处理后的数据形状: {processed_data.shape}")

    # 生成信号
    signals = momentum_strategy.generate_signals(processed_data)
    print(f"  信号序列形状: {signals.shape}")
    print(f"  买入信号数量: {(signals == 1).sum()}")
    print(f"  卖出信号数量: {(signals == -1).sum()}")

    # 2. 双动量策略
    print("\n2. 双动量策略:")
    dual_momentum_strategy = DualMomentumStrategy(
        absolute_lookback=20,
        relative_lookback=60,
        top_n=2
    )
    print(f"  策略名称: {dual_momentum_strategy.name}")

    # 3. 均值回归策略
    print("\n3. 均值回归策略:")
    mean_reversion_strategy = MeanReversionStrategy(
        lookback_period=20,
        zscore_threshold=2.0
    )
    print(f"  策略名称: {mean_reversion_strategy.name}")

    mean_rev_signals = mean_reversion_strategy.generate_signals(stock_data)
    print(f"  均值回归信号形状: {mean_rev_signals.shape}")

    # 4. 多因子策略
    print("\n4. 多因子策略:")
    multi_factor_strategy = MultiFactorStrategy(
        factors=["ma_10", "rsi_14", "macd"],
        weights=[0.4, 0.3, 0.3],
        top_n=2,
        rebalance_freq='M'
    )
    print(f"  策略名称: {multi_factor_strategy.name}")

    # 测试交易执行
    print("\n5. 测试交易执行:")
    capital = 1000000
    commission_rate = 0.0003

    for i in range(min(50, len(signals))):
        date = signals.index[i]
        signal_value = signals.iloc[i]
        price = processed_data['Close'].iloc[i]

        if signal_value == 1:  # BUY
            trade = momentum_strategy.execute_trade(
                symbol="AAPL",
                date=date,
                signal=1,  # BUY
                price=price,
                capital=capital,
                commission_rate=commission_rate
            )
            if trade:
                capital -= trade.value + trade.commission
                print(f"  买入交易: {trade.trade_date.date()} 价格={trade.price:.2f} 数量={trade.quantity}")

    print(f"  最终资金: {capital:.2f}")
    print(f"  总交易次数: {len(momentum_strategy.trade_history)}")

    return momentum_strategy


def example_backtest(strategy, stock_data):
    """示例：回测"""
    print("\n" + "=" * 80)
    print("示例4: 回测")
    print("=" * 80)

    # 初始化回测引擎
    backtester = Backtester(
        strategy=strategy,
        initial_capital=1000000,
        commission_rate=0.0003,
        slippage_rate=0.0001
    )

    print(f"回测引擎初始化完成")
    print(f"策略: {backtester.strategy.name}")
    print(f"初始资金: {backtester.initial_capital:,.2f}")
    print(f"手续费率: {backtester.commission_rate:.4f}")
    print(f"滑点率: {backtester.slippage_rate:.4f}")

    # 运行回测
    print("\n运行回测...")
    result = backtester.run(
        data=stock_data,
        start_date="2023-01-01",
        end_date="2023-12-31",
        mode=BacktestMode.SINGLE_STOCK
    )

    # 输出结果
    print(f"\n回测完成!")
    print(f"权益曲线长度: {len(result.equity_curve)}")
    print(f"收益率序列长度: {len(result.returns)}")
    print(f"交易记录数量: {len(result.trades) if not result.trades.empty else 0}")

    # 显示摘要
    summary = backtester.get_summary(result)
    print(f"\n回测摘要:")
    print(summary.to_string(index=False))

    # 显示性能指标
    print(f"\n性能指标:")
    for key, value in result.performance_metrics.items():
        if isinstance(value, float):
            if '率' in key or '比' in key:
                print(f"  {key}: {value:.4f}")
            elif '资金' in key or '权益' in key:
                print(f"  {key}: {value:,.2f}")
            else:
                print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    return result


def example_visualization(result, stock_data):
    """示例：可视化"""
    print("\n" + "=" * 80)
    print("示例5: 可视化")
    print("=" * 80)

    # 初始化可视化器
    plotter = Plotter()

    print("1. 绘制权益曲线图...")
    fig1 = plotter.plot_equity_curve(
        result.equity_curve,
        title="示例权益曲线",
        show=False
    )
    print(f"  权益曲线图创建成功")

    print("\n2. 绘制收益率分布图...")
    fig2 = plotter.plot_returns_distribution(
        result.returns,
        title="示例收益率分布",
        show=False
    )
    print(f"  收益率分布图创建成功")

    print("\n3. 绘制月度收益率热力图...")
    fig3 = plotter.plot_monthly_returns(
        result.returns,
        title="示例月度收益率",
        show=False
    )
    print(f"  月度收益率热力图创建成功")

    print("\n4. 绘制滚动指标图...")
    fig4 = plotter.plot_rolling_metrics(
        result.equity_curve,
        title="示例滚动指标",
        show=False
    )
    print(f"  滚动指标图创建成功")

    print("\n5. 绘制交易信号图...")
    if not result.trades.empty:
        price_data = stock_data['Close']
        fig5 = plotter.plot_trades(
            price_data,
            result.trades,
            title="示例交易信号",
            show=False
        )
        print(f"  交易信号图创建成功")

    print("\n6. 绘制性能摘要图...")
    fig6 = plotter.plot_performance_summary(
        result,
        title="示例性能摘要",
        show=False
    )
    print(f"  性能摘要图创建成功")

    print("\n7. 保存所有图表...")
    plotter.save_all_plots(
        result=result,
        price_data=stock_data['Close'] if not result.trades.empty else None,
        output_dir="./example_plots"
    )
    print(f"  图表已保存到: ./example_plots")


def example_web_app():
    """示例：Web应用"""
    print("\n" + "=" * 80)
    print("示例6: Web应用")
    print("=" * 80)

    print("""
    要启动Web应用，请运行以下命令:

        streamlit run web_app.py

    Web应用提供以下功能:

    1. 📊 数据配置
       - 输入股票代码
       - 选择日期范围
       - 选择数据频率

    2. 🎯 策略配置
       - 选择策略类型（动量、双动量、均值回归、多因子）
       - 调整策略参数
       - 选择技术指标和基本面因子

    3. 💰 回测配置
       - 设置初始资金
       - 调整手续费率和滑点率

    4. 📈 可视化分析
       - 权益曲线图
       - 收益率分布图
       - 月度收益率热力图
       - 滚动指标图
       - 交易信号图

    5. 📊 性能评估
       - 详细性能指标表格
       - 交易记录和持仓记录
       - 数据导出功能

    在浏览器中打开 http://localhost:8501 访问Web应用。
    """)


def main():
    """主函数"""
    print("量化选股软件使用示例")
    print("=" * 80)

    # 设置日志
    setup_logging()

    try:
        # 示例1: 数据获取
        stock_data, multi_data = example_data_fetcher()

        # 示例2: 因子计算
        factor_data = example_factor_calculator(stock_data)

        # 示例3: 策略
        strategy = example_strategies(stock_data, factor_data)

        # 示例4: 回测
        result = example_backtest(strategy, stock_data)

        # 示例5: 可视化
        example_visualization(result, stock_data)

        # 示例6: Web应用
        example_web_app()

        print("\n" + "=" * 80)
        print("所有示例运行完成!")
        print("=" * 80)

    except Exception as e:
        print(f"\n[ERROR] 运行示例时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()