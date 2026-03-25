#!/usr/bin/env python3
"""量化选股软件主程序"""
import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from src.data.data_fetcher_fixed import DataFetcherFixed as DataFetcher
except ImportError:
    from src.data.data_fetcher import DataFetcher

from src.factors.factor_calculator import FactorCalculator
from src.strategies.momentum import MomentumStrategy, DualMomentumStrategy, MeanReversionStrategy, MultiFactorStrategy
from src.backtest.backtester import Backtester, BacktestMode
from src.visualization.plotter import Plotter
from src.utils.config_loader import get_config_loader


def setup_logging(log_level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('quant_stock.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("量化选股软件启动")
    return logger


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="量化选股软件")

    # 数据获取参数
    parser.add_argument("--symbols", type=str, nargs="+", default=["AAPL", "MSFT", "GOOGL"],
                       help="股票代码列表")
    parser.add_argument("--start-date", type=str, default="2023-01-01",
                       help="开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2024-01-01",
                       help="结束日期 (YYYY-MM-DD)")
    parser.add_argument("--interval", type=str, default="1d",
                       choices=["1d", "1wk", "1mo"],
                       help="数据频率")

    # 策略参数
    parser.add_argument("--strategy", type=str, default="momentum",
                       choices=["momentum", "dual_momentum", "mean_reversion", "multi_factor"],
                       help="策略类型")
    parser.add_argument("--lookback", type=int, default=20,
                       help="回顾期")
    parser.add_argument("--holding", type=int, default=10,
                       help="持有期")
    parser.add_argument("--top-n", type=int, default=5,
                       help="选择前N只股票")

    # 回测参数
    parser.add_argument("--initial-capital", type=float, default=1000000,
                       help="初始资金")
    parser.add_argument("--commission", type=float, default=0.0003,
                       help="手续费率")
    parser.add_argument("--slippage", type=float, default=0.0001,
                       help="滑点率")

    # 输出参数
    parser.add_argument("--output-dir", type=str, default="./output",
                       help="输出目录")
    parser.add_argument("--plot", action="store_true",
                       help="生成图表")
    parser.add_argument("--verbose", action="store_true",
                       help="详细输出")

    # 其他参数
    parser.add_argument("--config", type=str, default=None,
                       help="配置文件路径")
    parser.add_argument("--cache", action="store_true",
                       help="启用缓存")
    parser.add_argument("--mock-data", action="store_true",
                       help="使用模拟数据（避免API限制）")

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()

    # 设置日志
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(log_level)

    # 加载配置
    config_loader = get_config_loader(args.config)
    logger.info(f"加载配置文件: {config_loader.config_path}")

    try:
        # 1. 获取数据
        logger.info("开始获取数据...")

        stock_data = {}

        # 使用修复版数据获取器
        fetcher = DataFetcher(cache_enabled=args.cache, use_mock_data=args.mock_data)

        # 获取多只股票数据
        for symbol in args.symbols:
            logger.info(f"获取股票数据: {symbol}")
            try:
                data = fetcher.get_stock_data(
                    symbol=symbol,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    interval=args.interval,
                    adjust=True
                )
                if data is not None and not data.empty:
                    stock_data[symbol] = data
                    logger.info(f"成功获取 {symbol} 数据，共 {len(data)} 条记录")
                else:
                    logger.warning(f"获取 {symbol} 数据失败或数据为空")
            except Exception as e:
                logger.error(f"获取 {symbol} 数据时出错: {e}")

        if not stock_data:
            logger.error("没有获取到任何股票数据")
            return

        # 2. 计算因子
        logger.info("开始计算因子...")
        calculator = FactorCalculator()

        # 准备因子数据
        factor_data = {}
        for symbol, data in stock_data.items():
            logger.info(f"计算 {symbol} 的因子...")
            try:
                factors = calculator.calculate_all_factors(data)
                factor_data[symbol] = factors
                logger.info(f"成功计算 {symbol} 的 {len(factors.columns)} 个因子")
            except Exception as e:
                logger.error(f"计算 {symbol} 因子时出错: {e}")

        # 3. 创建策略
        logger.info(f"创建策略: {args.strategy}")
        if args.strategy == "momentum":
            strategy = MomentumStrategy(
                lookback_period=args.lookback,
                holding_period=args.holding,
                top_n=args.top_n
            )
        elif args.strategy == "dual_momentum":
            strategy = DualMomentumStrategy(
                absolute_lookback=args.lookback,
                relative_lookback=args.lookback * 3,
                top_n=args.top_n
            )
        elif args.strategy == "mean_reversion":
            strategy = MeanReversionStrategy(
                lookback_period=args.lookback,
                zscore_threshold=2.0
            )
        elif args.strategy == "multi_factor":
            # 使用前几个因子
            all_factors = list(calculator.list_factors())[:5]
            weights = [1.0 / len(all_factors)] * len(all_factors)
            strategy = MultiFactorStrategy(
                factors=all_factors,
                weights=weights,
                top_n=args.top_n,
                rebalance_freq='M'
            )
        else:
            raise ValueError(f"不支持的策略类型: {args.strategy}")

        # 4. 准备回测数据
        logger.info("准备回测数据...")
        if args.strategy in ["momentum", "dual_momentum"]:
            # 多股票策略：使用价格数据
            backtest_data = {}
            for symbol, data in stock_data.items():
                backtest_data[symbol] = data['Close']
            backtest_data = pd.DataFrame(backtest_data)
            mode = BacktestMode.MULTI_STOCK
        else:
            # 单股票策略：使用第一只股票的数据
            first_symbol = list(stock_data.keys())[0]
            backtest_data = stock_data[first_symbol]
            mode = BacktestMode.SINGLE_STOCK

        # 5. 运行回测
        logger.info("开始回测...")
        backtester = Backtester(
            strategy=strategy,
            initial_capital=args.initial_capital,
            commission_rate=args.commission,
            slippage_rate=args.slippage
        )

        result = backtester.run(
            data=backtest_data,
            start_date=args.start_date,
            end_date=args.end_date,
            mode=mode
        )

        # 6. 输出结果
        logger.info("回测完成，输出结果...")

        # 创建输出目录
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存权益曲线
        equity_file = output_dir / "equity_curve.csv"
        result.equity_curve.to_csv(equity_file)
        logger.info(f"权益曲线已保存到: {equity_file}")

        # 保存交易记录
        if not result.trades.empty:
            trades_file = output_dir / "trades.csv"
            result.trades.to_csv(trades_file)
            logger.info(f"交易记录已保存到: {trades_file}")

        # 保存持仓记录
        if not result.positions.empty:
            positions_file = output_dir / "positions.csv"
            result.positions.to_csv(positions_file)
            logger.info(f"持仓记录已保存到: {positions_file}")

        # 保存性能指标
        metrics_file = output_dir / "performance_metrics.json"
        import json
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(result.performance_metrics, f, indent=2, ensure_ascii=False)
        logger.info(f"性能指标已保存到: {metrics_file}")

        # 7. 输出摘要
        print("\n" + "="*80)
        print("量化选股软件回测结果")
        print("="*80)

        summary = backtester.get_summary(result)
        print("\n回测摘要:")
        print(summary.to_string(index=False))

        print("\n性能指标:")
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

        # 8. 生成图表
        if args.plot:
            logger.info("生成图表...")
            plotter = Plotter()

            # 价格数据（用于交易信号图）
            price_data = None
            if mode == BacktestMode.SINGLE_STOCK:
                price_data = backtest_data['Close']
            elif mode == BacktestMode.MULTI_STOCK:
                # 使用第一只股票的价格
                first_symbol = list(stock_data.keys())[0]
                price_data = stock_data[first_symbol]['Close']

            plotter.save_all_plots(
                result=result,
                price_data=price_data,
                output_dir=str(output_dir / "plots")
            )

            print(f"\n图表已保存到: {output_dir / 'plots'}")

        print("\n" + "="*80)
        print("回测完成!")
        print("="*80)

    except Exception as e:
        logger.error(f"运行过程中出错: {e}", exc_info=True)
        sys.exit(1)


def run_example():
    """运行示例"""
    print("运行量化选股软件示例...")

    # 示例配置
    symbols = ["AAPL", "MSFT", "GOOGL"]
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    strategy = "momentum"

    print(f"股票代码: {symbols}")
    print(f"日期范围: {start_date} 到 {end_date}")
    print(f"策略类型: {strategy}")

    # 设置参数
    import sys
    sys.argv = [
        "main.py",
        "--symbols", *symbols,
        "--start-date", start_date,
        "--end-date", end_date,
        "--strategy", strategy,
        "--plot",
        "--verbose"
    ]

    main()


if __name__ == "__main__":
    # 如果没有命令行参数，运行示例
    if len(sys.argv) == 1:
        print("未提供命令行参数，运行示例...")
        run_example()
    else:
        main()