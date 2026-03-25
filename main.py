#!/usr/bin/env python3
"""量化选股软件主程序"""
import argparse
import logging
import sys
import os
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from src.data.data_fetcher_fixed import DataFetcherFixed as DataFetcher
except ImportError:
    from src.data.data_fetcher import DataFetcher

from src.factors.factor_calculator import FactorCalculator
from src.strategies.momentum import MomentumStrategy, DualMomentumStrategy, MeanReversionStrategy, MultiFactorStrategy
from src.backtest.backtester import Backtester, BacktestMode
from src.visualization.plotter import Plotter
from src.utils.config_loader import get_config_loader

# 股票分析模块导入
from src.analysis.stock_input import get_stocks, get_stock_list
from src.analysis.data_enhanced import EnhancedDataFetcher, StockData
from src.analysis.search_apis import SearchAPIManager
from src.analysis.llm_analyzer import LLMAnalyzer, AnalysisReport
from src.analysis.report_generator import ReportGenerator
from src.analysis.notifier import Notifier


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
    parser = argparse.ArgumentParser(
        description="量化选股软件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 回测模式
  python main.py backtest --symbols 600519 000858 --start-date 2023-01-01

  # 股票分析模式
  python main.py analyze --stocks 600519,000858
  python main.py analyze --stocks 600519 --output ./reports

  # 使用环境变量
  export STOCK_LIST="600519,000858,000333"
  python main.py analyze
        """
    )

    # 子命令
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # ========== 回测命令 ==========
    backtest_parser = subparsers.add_parser("backtest", help="运行回测")
    _add_backtest_args(backtest_parser)

    # ========== 分析命令 ==========
    analyze_parser = subparsers.add_parser("analyze", help="股票智能分析")
    _add_analyze_args(analyze_parser)

    # 兼容旧版本：如果没有子命令，使用回测模式
    parser.add_argument("--symbols", type=str, nargs="+",
                       help="股票代码列表 (兼容旧版)")
    parser.add_argument("--start-date", type=str, default="2023-01-01",
                       help="开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2024-01-01",
                       help="结束日期 (YYYY-MM-DD)")
    parser.add_argument("--interval", type=str, default="1d",
                       choices=["1d", "1wk", "1mo"],
                       help="数据频率")
    parser.add_argument("--strategy", type=str, default="momentum",
                       choices=["momentum", "dual_momentum", "mean_reversion", "multi_factor"],
                       help="策略类型")
    parser.add_argument("--lookback", type=int, default=20,
                       help="回顾期")
    parser.add_argument("--holding", type=int, default=10,
                       help="持有期")
    parser.add_argument("--top-n", type=int, default=5,
                       help="选择前N只股票")
    parser.add_argument("--initial-capital", type=float, default=1000000,
                       help="初始资金")
    parser.add_argument("--commission", type=float, default=0.0003,
                       help="手续费率")
    parser.add_argument("--slippage", type=float, default=0.0001,
                       help="滑点率")
    parser.add_argument("--output-dir", type=str, default="./output",
                       help="输出目录")
    parser.add_argument("--plot", action="store_true",
                       help="生成图表")
    parser.add_argument("--verbose", action="store_true",
                       help="详细输出")
    parser.add_argument("--config", type=str, default=None,
                       help="配置文件路径")
    parser.add_argument("--cache", action="store_true",
                       help="启用缓存")
    parser.add_argument("--mock-data", action="store_true",
                       help="使用模拟数据")

    return parser.parse_args()


def _add_backtest_args(parser):
    """添加回测命令参数"""
    parser.add_argument("--symbols", type=str, nargs="+", default=["600519", "000858"],
                       help="股票代码列表")
    parser.add_argument("--start-date", type=str, default="2023-01-01",
                       help="开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2024-01-01",
                       help="结束日期 (YYYY-MM-DD)")
    parser.add_argument("--interval", type=str, default="1d",
                       choices=["1d", "1wk", "1mo"],
                       help="数据频率")
    parser.add_argument("--strategy", type=str, default="momentum",
                       choices=["momentum", "dual_momentum", "mean_reversion", "multi_factor"],
                       help="策略类型")
    parser.add_argument("--lookback", type=int, default=20,
                       help="回顾期")
    parser.add_argument("--holding", type=int, default=10,
                       help="持有期")
    parser.add_argument("--top-n", type=int, default=5,
                       help="选择前N只股票")
    parser.add_argument("--initial-capital", type=float, default=1000000,
                       help="初始资金")
    parser.add_argument("--commission", type=float, default=0.0003,
                       help="手续费率")
    parser.add_argument("--slippage", type=float, default=0.0001,
                       help="滑点率")
    parser.add_argument("--output-dir", type=str, default="./output",
                       help="输出目录")
    parser.add_argument("--plot", action="store_true",
                       help="生成图表")
    parser.add_argument("--verbose", action="store_true",
                       help="详细输出")
    parser.add_argument("--config", type=str, default=None,
                       help="配置文件路径")
    parser.add_argument("--cache", action="store_true",
                       help="启用缓存")


def _add_analyze_args(parser):
    """添加分析命令参数"""
    parser.add_argument("--stocks", type=str,
                       help="股票代码列表 (逗号分隔或空格分隔)")
    parser.add_argument("--file", type=str,
                       help="股票列表文件路径 (每行一个代码)")
    parser.add_argument("--start-date", type=str, default=None,
                       help="历史数据开始日期")
    parser.add_argument("--end-date", type=str, default=None,
                       help="历史数据结束日期")
    parser.add_argument("--output", type=str, default="output/reports",
                       help="报告输出目录")
    parser.add_argument("--notify", action="store_true",
                       help="发送通知")
    parser.add_argument("--channels", type=str, nargs="+",
                       default=["email", "wechat"],
                       help="通知渠道")
    parser.add_argument("--no-search", action="store_true",
                       help="跳过搜索增强")
    parser.add_argument("--verbose", action="store_true",
                       help="详细输出")
    parser.add_argument("--config", type=str, default=None,
                       help="配置文件路径")


async def run_analyze(args, logger):
    """运行股票分析"""
    logger.info("开始股票智能分析...")

    # 1. 获取股票列表
    stocks = get_stock_list(
        stocks=args.stocks,
        file=args.file,
        env="STOCK_LIST"
    )

    if not stocks:
        logger.error("未获取到股票列表")
        return

    logger.info(f"分析股票: {stocks}")

    # 2. 初始化组件
    data_fetcher = EnhancedDataFetcher()
    search_manager = SearchAPIManager() if not args.no_search else None
    llm_analyzer = LLMAnalyzer()
    report_generator = ReportGenerator(output_dir=args.output)
    notifier = Notifier() if args.notify else None

    # 3. 日期范围
    start_date = args.start_date or (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    end_date = args.end_date or datetime.now().strftime("%Y-%m-%d")

    # 4. 分析每只股票
    analyses = []

    for symbol in stocks:
        logger.info(f"正在分析: {symbol}")

        try:
            # 获取数据
            stock_data = data_fetcher.get_full_stock_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                include_realtime=True,
                include_chip=True,
                include_institutional=False
            )

            # 搜索增强
            search_info = None
            if search_manager:
                logger.info(f"搜索 {symbol} 相关信息...")
                stock_name = stock_data.realtime.name if stock_data.realtime else ""
                search_info = await search_manager.search_stock_info(
                    symbol=symbol,
                    stock_name=stock_name
                )

            # LLM分析
            logger.info(f"LLM分析 {symbol}...")
            stock_name = stock_data.realtime.name if stock_data.realtime else ""
            analysis = await llm_analyzer.analyze_stock(
                stock_data=stock_data,
                search_info=search_info,
                stock_name=stock_name
            )

            # 生成报告
            report_files = report_generator.generate_single_report(
                symbol=symbol,
                stock_data=stock_data,
                analysis=analysis,
                stock_name=stock_name
            )

            logger.info(f"{symbol} 分析完成，评级: {analysis.rating}")
            logger.info(f"报告文件: {report_files}")

            analyses.append(analysis)

            # 发送通知
            if notifier and args.notify:
                summary = f"{analysis.technical_analysis[:500]}..." if analysis.technical_analysis else ""
                await notifier.send_analysis_report(
                    symbol=symbol,
                    stock_name=stock_name,
                    rating=analysis.rating,
                    summary=summary,
                    channels=args.channels
                )

        except Exception as e:
            logger.error(f"分析 {symbol} 失败: {e}")
            continue

    # 5. 生成汇总报告
    if len(analyses) > 1:
        summary_files = report_generator.generate_summary_report(analyses)
        logger.info(f"汇总报告: {summary_files}")

    # 6. 输出摘要
    print("\n" + "="*80)
    print("股票分析汇总")
    print("="*80)

    for analysis in analyses:
        print(f"\n{analysis.stock_name or analysis.symbol}:")
        print(f"  评级: {analysis.rating}")
        if analysis.target_price:
            print(f"  目标价: {analysis.target_price}")
        if analysis.stop_loss:
            print(f"  止损位: {analysis.stop_loss}")
        if analysis.take_profit:
            print(f"  止盈位: {analysis.take_profit}")

    print("\n" + "="*80)
    print(f"分析完成! 共分析 {len(analyses)} 只股票")
    print(f"报告目录: {args.output}")
    print("="*80)


def run_backtest(args, logger):
    """运行回测"""
    # 加载配置
    config_loader = get_config_loader(args.config if hasattr(args, 'config') else None)
    logger.info(f"加载配置文件: {config_loader.config_path}")

    try:
        # 1. 获取数据
        logger.info("开始获取数据...")

        stock_data = {}
        fetcher = DataFetcher(cache_enabled=getattr(args, 'cache', False), use_mock_data=getattr(args, 'mock_data', False))

        symbols = args.symbols if hasattr(args, 'symbols') else ["AAPL", "MSFT", "GOOGL"]
        start_date = args.start_date if hasattr(args, 'start_date') else "2023-01-01"
        end_date = args.end_date if hasattr(args, 'end_date') else "2024-01-01"
        interval = args.interval if hasattr(args, 'interval') else "1d"

        for symbol in symbols:
            logger.info(f"获取股票数据: {symbol}")
            try:
                data = fetcher.get_stock_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    interval=interval,
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
        strategy_type = args.strategy if hasattr(args, 'strategy') else "momentum"
        logger.info(f"创建策略: {strategy_type}")

        lookback = args.lookback if hasattr(args, 'lookback') else 20
        holding = args.holding if hasattr(args, 'holding') else 10
        top_n = args.top_n if hasattr(args, 'top_n') else 5

        if strategy_type == "momentum":
            strategy = MomentumStrategy(
                lookback_period=lookback,
                holding_period=holding,
                top_n=top_n
            )
        elif strategy_type == "dual_momentum":
            strategy = DualMomentumStrategy(
                absolute_lookback=lookback,
                relative_lookback=lookback * 3,
                top_n=top_n
            )
        elif strategy_type == "mean_reversion":
            strategy = MeanReversionStrategy(
                lookback_period=lookback,
                zscore_threshold=2.0
            )
        elif strategy_type == "multi_factor":
            all_factors = list(calculator.list_factors())[:5]
            weights = [1.0 / len(all_factors)] * len(all_factors)
            strategy = MultiFactorStrategy(
                factors=all_factors,
                weights=weights,
                top_n=top_n,
                rebalance_freq='M'
            )
        else:
            raise ValueError(f"不支持的策略类型: {strategy_type}")

        # 4. 准备回测数据
        logger.info("准备回测数据...")
        if strategy_type in ["momentum", "dual_momentum"]:
            backtest_data = {}
            for symbol, data in stock_data.items():
                backtest_data[symbol] = data['Close']
            backtest_data = pd.DataFrame(backtest_data)
            mode = BacktestMode.MULTI_STOCK
        else:
            first_symbol = list(stock_data.keys())[0]
            backtest_data = stock_data[first_symbol]
            mode = BacktestMode.SINGLE_STOCK

        # 5. 运行回测
        logger.info("开始回测...")

        initial_capital = args.initial_capital if hasattr(args, 'initial_capital') else 1000000
        commission = args.commission if hasattr(args, 'commission') else 0.0003
        slippage = args.slippage if hasattr(args, 'slippage') else 0.0001

        backtester = Backtester(
            strategy=strategy,
            initial_capital=initial_capital,
            commission_rate=commission,
            slippage_rate=slippage
        )

        result = backtester.run(
            data=backtest_data,
            start_date=start_date,
            end_date=end_date,
            mode=mode
        )

        # 6. 输出结果
        logger.info("回测完成，输出结果...")

        output_dir = Path(args.output_dir if hasattr(args, 'output_dir') else "./output")
        output_dir.mkdir(parents=True, exist_ok=True)

        equity_file = output_dir / "equity_curve.csv"
        result.equity_curve.to_csv(equity_file)
        logger.info(f"权益曲线已保存到: {equity_file}")

        if not result.trades.empty:
            trades_file = output_dir / "trades.csv"
            result.trades.to_csv(trades_file)
            logger.info(f"交易记录已保存到: {trades_file}")

        if not result.positions.empty:
            positions_file = output_dir / "positions.csv"
            result.positions.to_csv(positions_file)
            logger.info(f"持仓记录已保存到: {positions_file}")

        import json
        metrics_file = output_dir / "performance_metrics.json"
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
        if getattr(args, 'plot', False):
            logger.info("生成图表...")
            plotter = Plotter()

            price_data = None
            if mode == BacktestMode.SINGLE_STOCK:
                price_data = backtest_data['Close']
            elif mode == BacktestMode.MULTI_STOCK:
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


def main():
    """主函数"""
    args = parse_arguments()

    # 设置日志
    verbose = getattr(args, 'verbose', False)
    log_level = "DEBUG" if verbose else "INFO"
    logger = setup_logging(log_level)

    # 根据命令分发
    if args.command == "analyze":
        asyncio.run(run_analyze(args, logger))
    elif args.command == "backtest":
        run_backtest(args, logger)
    elif args.symbols:
        # 兼容旧版本：直接传入symbols参数时运行回测
        run_backtest(args, logger)
    else:
        # 默认显示帮助
        print("请指定命令: backtest 或 analyze")
        print("使用 --help 查看帮助信息")
        sys.exit(1)


def run_example():
    """运行示例"""
    print("运行量化选股软件示例...")

    # 示例配置
    symbols = ["600519", "000858"]
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    strategy = "momentum"

    print(f"股票代码: {symbols}")
    print(f"日期范围: {start_date} 到 {end_date}")
    print(f"策略类型: {strategy}")

    # 设置参数
    sys.argv = [
        "main.py", "backtest",
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
