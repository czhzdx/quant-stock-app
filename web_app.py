"""量化选股软件Web界面"""
import sys
import os
import asyncio
from pathlib import Path

# 确保项目根目录在 Python 路径中
root_dir = Path(__file__).parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from src.data.data_fetcher import DataFetcher
from src.factors.factor_calculator import FactorCalculator, FactorType
from src.strategies.momentum import MomentumStrategy, DualMomentumStrategy, MeanReversionStrategy, MultiFactorStrategy
from src.backtest.backtester import Backtester, BacktestMode
from src.visualization.plotter import Plotter
from src.utils.config_loader import get_config_loader

# 股票分析模块
from src.analysis.data_enhanced import EnhancedDataFetcher
from src.analysis.search_apis import SearchAPIManager
from src.analysis.llm_analyzer import LLMAnalyzer
from src.analysis.report_generator import ReportGenerator
from src.analysis.notifier import Notifier

# 页面配置
st.set_page_config(
    page_title="量化选股软件",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #424242;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #1E88E5;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin-bottom: 1rem;
    }
    .success-text {
        color: #4CAF50;
        font-weight: bold;
    }
    .warning-text {
        color: #FF9800;
        font-weight: bold;
    }
    .error-text {
        color: #F44336;
        font-weight: bold;
    }
    .rating-strong-buy {
        color: #e91e63;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .rating-buy {
        color: #4CAF50;
        font-weight: bold;
    }
    .rating-neutral {
        color: #FF9800;
        font-weight: bold;
    }
    .rating-sell {
        color: #F44336;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# 标题
st.markdown('<h1 class="main-header">📈 量化选股软件</h1>', unsafe_allow_html=True)

# 创建标签页
tab1, tab2 = st.tabs(["🔄 策略回测", "🤖 智能分析"])


# ========== 标签页1: 策略回测 ==========
with tab1:
    # 侧边栏配置
    with st.sidebar:
        st.markdown("## ⚙️ 回测配置")

        # 数据获取配置
        st.markdown("### 📊 数据配置")
        st.markdown("股票代码格式：A股(如600519)、港股(如00700)、美股(如AAPL)")

        symbols_input = st.text_area(
            "股票代码 (每行一个或逗号分隔)",
            value="600519\n000858\n000333\n601318\n600036",
            height=100,
            key="backtest_symbols"
        )

        # 解析股票代码
        symbols = []
        for line in symbols_input.split('\n'):
            line = line.strip()
            if ',' in line:
                symbols.extend([s.strip() for s in line.split(',') if s.strip()])
            elif line:
                symbols.append(line)

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "开始日期",
                value=datetime.now() - timedelta(days=365),
                max_value=datetime.now(),
                key="backtest_start"
            )
        with col2:
            end_date = st.date_input(
                "结束日期",
                value=datetime.now(),
                max_value=datetime.now(),
                key="backtest_end"
            )

        interval = st.selectbox(
            "数据频率",
            options=["1d", "1wk", "1mo"],
            index=0,
            key="backtest_interval"
        )

        # 策略配置
        st.markdown("### 🎯 策略配置")
        strategy_type = st.selectbox(
            "策略类型",
            options=["动量策略", "双动量策略", "均值回归策略", "多因子策略"],
            index=0,
            key="backtest_strategy"
        )

        if strategy_type == "动量策略":
            lookback = st.slider("回顾期 (天)", 5, 250, 20, key="bt_lookback")
            holding = st.slider("持有期 (天)", 1, 60, 10, key="bt_holding")
            top_n = st.slider("选择前N只股票", 1, 20, 5, key="bt_top_n")
        elif strategy_type == "双动量策略":
            abs_lookback = st.slider("绝对动量回顾期 (天)", 5, 100, 20, key="bt_abs_lookback")
            rel_lookback = st.slider("相对动量回顾期 (天)", 20, 250, 60, key="bt_rel_lookback")
            top_n = st.slider("选择前N只股票", 1, 20, 5, key="bt_top_n2")
        elif strategy_type == "均值回归策略":
            lookback = st.slider("回顾期 (天)", 5, 250, 20, key="bt_lookback2")
            zscore_threshold = st.slider("Z分数阈值", 1.0, 3.0, 2.0, 0.1, key="bt_zscore")
        else:
            top_n = st.slider("选择前N只股票", 1, 20, 5, key="bt_top_n3")
            rebalance_freq = st.selectbox(
                "再平衡频率",
                options=["每日", "每周", "每月"],
                index=2,
                key="bt_rebalance"
            )

        # 回测配置
        st.markdown("### 💰 回测配置")
        initial_capital = st.number_input(
            "初始资金",
            min_value=10000,
            max_value=100000000,
            value=1000000,
            step=10000,
            key="bt_capital"
        )

        col1, col2 = st.columns(2)
        with col1:
            commission_rate = st.number_input(
                "手续费率 (%)",
                min_value=0.0,
                max_value=1.0,
                value=0.03,
                step=0.01,
                format="%.2f",
                key="bt_commission"
            ) / 100

        with col2:
            slippage_rate = st.number_input(
                "滑点率 (%)",
                min_value=0.0,
                max_value=1.0,
                value=0.01,
                step=0.01,
                format="%.2f",
                key="bt_slippage"
            ) / 100

        # 因子配置
        st.markdown("### 🔢 因子配置")
        factor_calculator = FactorCalculator()

        technical_factors = factor_calculator.list_factors(FactorType.TECHNICAL)
        default_technical = ["ma_20", "macd"]
        rsi_factors = [f for f in technical_factors if f.startswith("rsi_")]
        if rsi_factors:
            default_technical.append(rsi_factors[0])
        selected_technical = st.multiselect(
            "技术指标",
            options=technical_factors,
            default=default_technical,
            key="bt_technical"
        )

        # 执行按钮
        st.markdown("---")
        run_backtest = st.button("🚀 开始回测", type="primary", use_container_width=True, key="btn_backtest")

    # 回测主逻辑
    if run_backtest and symbols:
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()

            # 1. 获取数据
            status_text.text("📥 正在获取股票数据...")
            fetcher = DataFetcher(cache_enabled=True)

            stock_data = {}
            for i, symbol in enumerate(symbols):
                try:
                    data = fetcher.get_stock_data(
                        symbol=symbol,
                        start_date=start_date.strftime("%Y-%m-%d"),
                        end_date=end_date.strftime("%Y-%m-%d"),
                        interval=interval,
                        adjust=True
                    )
                    if data is not None and not data.empty:
                        stock_data[symbol] = data
                        st.success(f"✅ 成功获取 {symbol} 数据 ({len(data)} 条记录)")
                except Exception as e:
                    st.error(f"❌ 获取 {symbol} 数据时出错: {str(e)}")
                progress_bar.progress((i + 1) / len(symbols) * 0.3)

            if not stock_data:
                st.error("❌ 没有获取到任何股票数据")
                st.stop()

            # 2. 计算因子
            status_text.text("🔢 正在计算因子...")
            calculator = FactorCalculator()
            factor_data = {}
            selected_factors = selected_technical

            for i, (symbol, data) in enumerate(stock_data.items()):
                if selected_factors:
                    try:
                        factors = calculator.calculate_multiple_factors(data, selected_factors)
                        factor_data[symbol] = factors
                    except Exception as e:
                        st.error(f"❌ 计算 {symbol} 因子时出错: {str(e)}")
                progress_bar.progress(0.3 + (i + 1) / len(stock_data) * 0.2)

            # 3. 创建策略
            status_text.text("🎯 正在创建策略...")
            if strategy_type == "动量策略":
                strategy = MomentumStrategy(lookback_period=lookback, holding_period=holding, top_n=top_n)
            elif strategy_type == "双动量策略":
                strategy = DualMomentumStrategy(absolute_lookback=abs_lookback, relative_lookback=rel_lookback, top_n=top_n)
            elif strategy_type == "均值回归策略":
                strategy = MeanReversionStrategy(lookback_period=lookback, zscore_threshold=zscore_threshold)
            else:
                freq_map = {"每日": "D", "每周": "W", "每月": "M"}
                strategy = MultiFactorStrategy(
                    factors=selected_factors,
                    weights=[1.0/len(selected_factors)] * len(selected_factors) if selected_factors else [],
                    top_n=top_n,
                    rebalance_freq=freq_map[rebalance_freq]
                )

            progress_bar.progress(0.6)

            # 4. 准备回测数据
            status_text.text("📊 正在准备回测数据...")
            if strategy_type in ["动量策略", "双动量策略"]:
                backtest_data = pd.DataFrame({s: d['Close'] for s, d in stock_data.items()})
                mode = BacktestMode.MULTI_STOCK
            else:
                first_symbol = list(stock_data.keys())[0]
                backtest_data = stock_data[first_symbol]
                mode = BacktestMode.SINGLE_STOCK

            progress_bar.progress(0.7)

            # 5. 运行回测
            status_text.text("⚡ 正在运行回测...")
            backtester = Backtester(
                strategy=strategy,
                initial_capital=initial_capital,
                commission_rate=commission_rate,
                slippage_rate=slippage_rate
            )

            result = backtester.run(
                data=backtest_data,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                mode=mode
            )

            progress_bar.progress(0.9)
            status_text.text("📈 正在生成报告...")
            progress_bar.empty()
            status_text.empty()

            # 6. 显示结果
            st.markdown('<h2 class="section-header">📊 回测结果摘要</h2>', unsafe_allow_html=True)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                total_return = result.performance_metrics.get('总收益率%', 0)
                return_color = "success-text" if total_return > 0 else "error-text"
                st.metric("总收益率", f"{total_return:.2f}%")
            with col2:
                st.metric("年化收益率", f"{result.performance_metrics.get('年化收益率%', 0):.2f}%")
            with col3:
                sharpe = result.performance_metrics.get('夏普比率', 0)
                st.metric("夏普比率", f"{sharpe:.3f}")
            with col4:
                st.metric("最大回撤", f"{result.performance_metrics.get('最大回撤%', 0):.2f}%")

            # 图表
            st.markdown('<h3 class="section-header">📈 可视化图表</h3>', unsafe_allow_html=True)
            plotter = Plotter()

            fig1 = plotter.plot_equity_curve(result.equity_curve, result.benchmark_returns, title="策略权益曲线", show=False)
            st.plotly_chart(fig1, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                fig2 = plotter.plot_returns_distribution(result.returns, title="收益率分布", show=False)
                st.plotly_chart(fig2, use_container_width=True)
            with col2:
                fig3 = plotter.plot_monthly_returns(result.returns, title="月度收益率", show=False)
                st.plotly_chart(fig3, use_container_width=True)

            st.success("✅ 回测完成!")

        except Exception as e:
            st.error(f"❌ 运行出错: {str(e)}")
            import traceback
            with st.expander("查看错误详情"):
                st.code(traceback.format_exc())
    else:
        st.info("👈 请在左侧配置参数后点击「开始回测」")


# ========== 标签页2: 智能分析 ==========
with tab2:
    # 分析配置
    with st.sidebar:
        st.markdown("## 🤖 分析配置")

        st.markdown("### 📊 股票选择")
        analyze_symbols = st.text_input(
            "股票代码 (逗号分隔)",
            value="600519,000858,000333",
            key="analyze_symbols"
        )

        # 解析股票代码
        analyze_stocks = [s.strip() for s in analyze_symbols.split(',') if s.strip()]

        st.markdown("### 📅 数据范围")
        col1, col2 = st.columns(2)
        with col1:
            analyze_start = st.date_input(
                "开始日期",
                value=datetime.now() - timedelta(days=180),
                key="analyze_start"
            )
        with col2:
            analyze_end = st.date_input(
                "结束日期",
                value=datetime.now(),
                key="analyze_end"
            )

        st.markdown("### 🔧 分析选项")
        enable_search = st.checkbox("启用搜索增强", value=True, key="analyze_search")
        enable_chip = st.checkbox("获取筹码数据", value=True, key="analyze_chip")
        enable_realtime = st.checkbox("获取实时行情", value=True, key="analyze_realtime")

        st.markdown("### 📬 通知设置")
        enable_notify = st.checkbox("发送通知", value=False, key="analyze_notify")
        notify_channels = st.multiselect(
            "通知渠道",
            options=["email", "wechat"],
            default=["email"],
            key="analyze_channels"
        )

        st.markdown("---")
        run_analyze = st.button("🤖 开始智能分析", type="primary", use_container_width=True, key="btn_analyze")

    # 分析主逻辑
    if run_analyze and analyze_stocks:
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()

            # 初始化组件
            status_text.text("🔧 初始化分析组件...")
            data_fetcher = EnhancedDataFetcher()
            search_manager = SearchAPIManager() if enable_search else None
            llm_analyzer = LLMAnalyzer()
            report_generator = ReportGenerator()
            notifier = Notifier() if enable_notify else None

            progress_bar.progress(0.1)

            # 存储分析结果
            analyses_results = []

            # 分析每只股票
            for i, symbol in enumerate(analyze_stocks):
                status_text.text(f"📊 正在分析 {symbol}...")

                # 获取数据
                with st.spinner(f"获取 {symbol} 数据..."):
                    stock_data = data_fetcher.get_full_stock_data(
                        symbol=symbol,
                        start_date=analyze_start.strftime("%Y-%m-%d"),
                        end_date=analyze_end.strftime("%Y-%m-%d"),
                        include_realtime=enable_realtime,
                        include_chip=enable_chip,
                        include_institutional=False
                    )

                # 显示基本信息
                if stock_data.realtime:
                    rt = stock_data.realtime
                    st.markdown(f"### 📈 {rt.name} ({symbol})")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("最新价", f"¥{rt.price:.2f}", f"{rt.change_pct:.2f}%")
                    with col2:
                        st.metric("成交量", f"{rt.volume:,}")
                    with col3:
                        st.metric("换手率", f"{rt.turnover_rate:.2f}%")
                    with col4:
                        st.metric("成交额", f"¥{rt.amount:,.0f}")
                    stock_name = rt.name
                else:
                    st.markdown(f"### 📈 {symbol}")
                    stock_name = symbol

                # 搜索增强
                search_info = None
                if search_manager and enable_search:
                    status_text.text(f"🔍 搜索 {symbol} 相关信息...")
                    with st.spinner("搜索相关资讯..."):
                        search_info = asyncio.run(search_manager.search_stock_info(
                            symbol=symbol,
                            stock_name=stock_name
                        ))

                    # 显示搜索结果
                    if search_info and search_info.news:
                        with st.expander("📰 相关新闻", expanded=False):
                            for news in search_info.news[:5]:
                                st.markdown(f"- **{news.title}**")
                                if news.snippet:
                                    st.markdown(f"  {news.snippet[:100]}...")

                progress_bar.progress(0.3 + (i / len(analyze_stocks)) * 0.3)

                # LLM分析
                status_text.text(f"🤖 AI分析 {symbol}...")
                with st.spinner("正在进行AI深度分析，请稍候..."):
                    analysis = asyncio.run(llm_analyzer.analyze_stock(
                        stock_data=stock_data,
                        search_info=search_info,
                        stock_name=stock_name
                    ))

                progress_bar.progress(0.6 + (i / len(analyze_stocks)) * 0.3)

                # 显示分析结果
                st.markdown("---")

                # 评级显示
                rating_colors = {
                    "强烈推荐": "rating-strong-buy",
                    "推荐": "rating-buy",
                    "中性": "rating-neutral",
                    "不推荐": "rating-sell",
                    "强烈不推荐": "rating-sell"
                }
                rating_class = rating_colors.get(analysis.rating, "rating-neutral")

                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.markdown(f"**投资评级**: <span class='{rating_class}'>{analysis.rating}</span>", unsafe_allow_html=True)
                with col2:
                    if analysis.target_price:
                        st.metric("目标价", f"¥{analysis.target_price:.2f}")
                with col3:
                    if analysis.stop_loss:
                        st.metric("止损位", f"¥{analysis.stop_loss:.2f}")

                # 详细分析
                tab_a, tab_b, tab_c, tab_d = st.tabs(["基本面", "技术面", "筹码分析", "操作建议"])

                with tab_a:
                    if analysis.fundamental_analysis:
                        st.markdown(analysis.fundamental_analysis)
                    else:
                        st.info("暂无基本面分析")

                with tab_b:
                    if analysis.technical_analysis:
                        st.markdown(analysis.technical_analysis)
                    else:
                        st.info("暂无技术面分析")

                with tab_c:
                    if analysis.chip_analysis:
                        st.markdown(analysis.chip_analysis)
                    elif stock_data.chip_distribution:
                        chip = stock_data.chip_distribution
                        st.metric("获利盘比例", f"{chip.profit_ratio:.2f}%")
                        st.metric("平均成本", f"¥{chip.avg_cost:.2f}")
                    else:
                        st.info("暂无筹码数据")

                with tab_d:
                    if analysis.operation_advice:
                        st.markdown(analysis.operation_advice)
                    else:
                        st.info("暂无操作建议")

                # 风险提示
                if analysis.risks:
                    st.warning("⚠️ **风险提示**: " + ", ".join(analysis.risks))

                analyses_results.append(analysis)

                # 发送通知
                if notifier and enable_notify:
                    summary = analysis.technical_analysis[:500] if analysis.technical_analysis else ""
                    asyncio.run(notifier.send_analysis_report(
                        symbol=symbol,
                        stock_name=stock_name,
                        rating=analysis.rating,
                        summary=summary,
                        channels=notify_channels
                    ))

            progress_bar.progress(1.0)
            progress_bar.empty()
            status_text.empty()

            # 汇总结果
            if len(analyses_results) > 1:
                st.markdown("---")
                st.markdown('<h2 class="section-header">📊 分析汇总</h2>', unsafe_allow_html=True)

                # 汇总表格
                summary_data = []
                for a in analyses_results:
                    summary_data.append({
                        "股票": f"{a.stock_name} ({a.symbol})",
                        "评级": a.rating,
                        "目标价": f"¥{a.target_price:.2f}" if a.target_price else "-",
                        "止损位": f"¥{a.stop_loss:.2f}" if a.stop_loss else "-",
                        "模型": a.model_used
                    })

                st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

                # 生成汇总报告
                report_files = report_generator.generate_summary_report(analyses_results)
                st.success(f"✅ 汇总报告已生成: {report_files}")

                # 下载按钮
                if "markdown" in report_files:
                    with open(report_files["markdown"], "r", encoding="utf-8") as f:
                        st.download_button(
                            label="📥 下载汇总报告 (Markdown)",
                            data=f.read(),
                            file_name="analysis_summary.md",
                            mime="text/markdown"
                        )

            st.success("✅ 智能分析完成!")

        except Exception as e:
            st.error(f"❌ 分析出错: {str(e)}")
            import traceback
            with st.expander("查看错误详情"):
                st.code(traceback.format_exc())
    else:
        st.info("👈 请在左侧输入股票代码后点击「开始智能分析」")

        # 显示功能说明
        st.markdown("""
        ### 🤖 智能分析功能

        本功能利用大语言模型(LLM)对股票进行深度分析，包括：

        - **基本面分析**: 公司业务、行业地位、主要风险
        - **技术面分析**: 趋势判断、支撑压力位、成交量分析
        - **筹码分析**: 筹码集中度、获利盘、主力动向
        - **消息面分析**: 新闻解读、公告影响
        - **投资建议**: 评级、目标价、止损位

        ### 📋 使用说明

        1. 在左侧输入股票代码（逗号分隔）
        2. 选择数据范围和分析选项
        3. 点击「开始智能分析」
        4. 等待AI分析完成

        ### ⚙️ 配置要求

        需要配置API密钥（环境变量或配置文件）:
        - `OPENAI_API_KEY`: OpenAI API密钥（必需）
        - `TAVILY_API_KEY`: 搜索增强API密钥（可选）
        """)


# 页脚
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>📈 量化选股软件 v2.0 | 策略回测 + 智能分析</p>
    <p>⚠️ 风险提示：本软件仅供学习和研究使用，不构成投资建议</p>
</div>
""", unsafe_allow_html=True)
