"""量化选股软件Web界面"""
import sys
from pathlib import Path

# 确保项目根目录在 Python 路径中
root_dir = Path(__file__).parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

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
</style>
""", unsafe_allow_html=True)

# 标题
st.markdown('<h1 class="main-header">📈 量化选股软件</h1>', unsafe_allow_html=True)

# 侧边栏
with st.sidebar:
    st.markdown("## ⚙️ 配置参数")

    # 数据获取配置
    st.markdown("### 📊 数据配置")
    st.markdown("股票代码格式：A股(如600519)、港股(如00700)、美股(如AAPL)")

    symbols_input = st.text_area(
        "股票代码 (每行一个或逗号分隔)",
        value="600519\n000858\n000333\n601318\n600036",
        height=100
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
            max_value=datetime.now()
        )
    with col2:
        end_date = st.date_input(
            "结束日期",
            value=datetime.now(),
            max_value=datetime.now()
        )

    interval = st.selectbox(
        "数据频率",
        options=["1d", "1wk", "1mo"],
        index=0
    )

    # 策略配置
    st.markdown("### 🎯 策略配置")
    strategy_type = st.selectbox(
        "策略类型",
        options=["动量策略", "双动量策略", "均值回归策略", "多因子策略"],
        index=0
    )

    if strategy_type == "动量策略":
        lookback = st.slider("回顾期 (天)", 5, 250, 20)
        holding = st.slider("持有期 (天)", 1, 60, 10)
        top_n = st.slider("选择前N只股票", 1, 20, 5)
    elif strategy_type == "双动量策略":
        abs_lookback = st.slider("绝对动量回顾期 (天)", 5, 100, 20)
        rel_lookback = st.slider("相对动量回顾期 (天)", 20, 250, 60)
        top_n = st.slider("选择前N只股票", 1, 20, 5)
    elif strategy_type == "均值回归策略":
        lookback = st.slider("回顾期 (天)", 5, 250, 20)
        zscore_threshold = st.slider("Z分数阈值", 1.0, 3.0, 2.0, 0.1)
    else:  # 多因子策略
        top_n = st.slider("选择前N只股票", 1, 20, 5)
        rebalance_freq = st.selectbox(
            "再平衡频率",
            options=["每日", "每周", "每月"],
            index=2
        )

    # 回测配置
    st.markdown("### 💰 回测配置")
    initial_capital = st.number_input(
        "初始资金",
        min_value=10000,
        max_value=100000000,
        value=1000000,
        step=10000
    )

    col1, col2 = st.columns(2)
    with col1:
        commission_rate = st.number_input(
            "手续费率 (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.03,
            step=0.01,
            format="%.2f"
        ) / 100

    with col2:
        slippage_rate = st.number_input(
            "滑点率 (%)",
            min_value=0.0,
            max_value=1.0,
            value=0.01,
            step=0.01,
            format="%.2f"
        ) / 100

    # 因子配置
    st.markdown("### 🔢 因子配置")
    factor_calculator = FactorCalculator()

    technical_factors = factor_calculator.list_factors(FactorType.TECHNICAL)
    # 使用第一个可用的因子作为默认值，避免默认值不在选项中的错误
    default_technical = ["ma_20", "macd"]
    # 如果有RSI因子，添加一个
    rsi_factors = [f for f in technical_factors if f.startswith("rsi_")]
    if rsi_factors:
        default_technical.append(rsi_factors[0])
    selected_technical = st.multiselect(
        "技术指标",
        options=technical_factors,
        default=default_technical
    )

    fundamental_factors = factor_calculator.list_factors(FactorType.FUNDAMENTAL)
    selected_fundamental = st.multiselect(
        "基本面因子",
        options=fundamental_factors,
        default=[]
    )

    # 执行按钮
    st.markdown("---")
    run_button = st.button("🚀 开始回测", type="primary", use_container_width=True)

    # 重置按钮
    if st.button("🔄 重置", use_container_width=True):
        st.rerun()

# 主内容区域
if run_button and symbols:
    try:
        # 显示进度
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
                else:
                    st.warning(f"⚠️ 获取 {symbol} 数据失败或数据为空")
            except Exception as e:
                st.error(f"❌ 获取 {symbol} 数据时出错: {str(e)}")

            progress_bar.progress((i + 1) / len(symbols) * 0.3)

        if not stock_data:
            st.error("❌ 没有获取到任何股票数据，请检查股票代码")
            st.stop()

        # 2. 计算因子
        status_text.text("🔢 正在计算因子...")
        calculator = FactorCalculator()

        factor_data = {}
        selected_factors = selected_technical + selected_fundamental

        for i, (symbol, data) in enumerate(stock_data.items()):
            if selected_factors:
                try:
                    factors = calculator.calculate_multiple_factors(data, selected_factors)
                    factor_data[symbol] = factors
                    st.success(f"✅ 成功计算 {symbol} 的 {len(factors.columns)} 个因子")
                except Exception as e:
                    st.error(f"❌ 计算 {symbol} 因子时出错: {str(e)}")

            progress_bar.progress(0.3 + (i + 1) / len(stock_data) * 0.2)

        # 3. 创建策略
        status_text.text("🎯 正在创建策略...")

        if strategy_type == "动量策略":
            strategy = MomentumStrategy(
                lookback_period=lookback,
                holding_period=holding,
                top_n=top_n
            )
        elif strategy_type == "双动量策略":
            strategy = DualMomentumStrategy(
                absolute_lookback=abs_lookback,
                relative_lookback=rel_lookback,
                top_n=top_n
            )
        elif strategy_type == "均值回归策略":
            strategy = MeanReversionStrategy(
                lookback_period=lookback,
                zscore_threshold=zscore_threshold
            )
        else:  # 多因子策略
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
            # 多股票策略
            backtest_data = {}
            for symbol, data in stock_data.items():
                backtest_data[symbol] = data['Close']
            backtest_data = pd.DataFrame(backtest_data)
            mode = BacktestMode.MULTI_STOCK
        else:
            # 单股票策略：使用第一只股票
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

        # 6. 显示结果
        status_text.text("📈 正在生成报告...")

        # 清除进度条
        progress_bar.empty()
        status_text.empty()

        # 显示摘要
        st.markdown('<h2 class="section-header">📊 回测结果摘要</h2>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_return = result.performance_metrics.get('总收益率%', 0)
            return_color = "success-text" if total_return > 0 else "error-text"
            st.markdown(f"""
            <div class="metric-card">
                <h3>总收益率</h3>
                <p class="{return_color}">{total_return:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            annual_return = result.performance_metrics.get('年化收益率%', 0)
            st.markdown(f"""
            <div class="metric-card">
                <h3>年化收益率</h3>
                <p>{annual_return:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            sharpe_ratio = result.performance_metrics.get('夏普比率', 0)
            sharpe_color = "success-text" if sharpe_ratio > 1 else "warning-text" if sharpe_ratio > 0 else "error-text"
            st.markdown(f"""
            <div class="metric-card">
                <h3>夏普比率</h3>
                <p class="{sharpe_color}">{sharpe_ratio:.3f}</p>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            max_drawdown = result.performance_metrics.get('最大回撤%', 0)
            drawdown_color = "error-text" if max_drawdown < -20 else "warning-text" if max_drawdown < -10 else "success-text"
            st.markdown(f"""
            <div class="metric-card">
                <h3>最大回撤</h3>
                <p class="{drawdown_color}">{max_drawdown:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

        # 显示详细指标
        st.markdown('<h3 class="section-header">📈 详细性能指标</h3>', unsafe_allow_html=True)

        metrics_df = pd.DataFrame([
            ["初始资金", f"¥{result.performance_metrics.get('初始资金', 0):,.2f}"],
            ["最终权益", f"¥{result.performance_metrics.get('最终权益', 0):,.2f}"],
            ["总收益率", f"{result.performance_metrics.get('总收益率%', 0):.2f}%"],
            ["年化收益率", f"{result.performance_metrics.get('年化收益率%', 0):.2f}%"],
            ["年化波动率", f"{result.performance_metrics.get('年化波动率%', 0):.2f}%"],
            ["夏普比率", f"{result.performance_metrics.get('夏普比率', 0):.3f}"],
            ["最大回撤", f"{result.performance_metrics.get('最大回撤%', 0):.2f}%"],
            ["索提诺比率", f"{result.performance_metrics.get('索提诺比率', 0):.3f}"],
            ["卡尔玛比率", f"{result.performance_metrics.get('卡尔玛比率', 0):.3f}"],
            ["胜率", f"{result.performance_metrics.get('胜率%', 0):.2f}%"],
            ["盈亏比", f"{result.performance_metrics.get('盈亏比', 0):.2f}"],
            ["总交易次数", f"{result.performance_metrics.get('总交易次数', 0)}"],
            ["盈利交易次数", f"{result.performance_metrics.get('winning_trades', 0)}"],
            ["亏损交易次数", f"{result.performance_metrics.get('losing_trades', 0)}"],
            ["Alpha", f"{result.performance_metrics.get('Alpha', 0):.3f}"],
            ["Beta", f"{result.performance_metrics.get('Beta', 0):.3f}"],
            ["交易天数", f"{result.performance_metrics.get('交易天数', 0)}"]
        ], columns=["指标", "值"])

        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        # 显示图表
        st.markdown('<h3 class="section-header">📊 可视化图表</h3>', unsafe_allow_html=True)

        plotter = Plotter()

        # 权益曲线
        fig1 = plotter.plot_equity_curve(
            result.equity_curve,
            result.benchmark_returns,
            title="策略权益曲线",
            show=False
        )
        st.plotly_chart(fig1, use_container_width=True)

        # 收益率分布
        fig2 = plotter.plot_returns_distribution(
            result.returns,
            title="收益率分布",
            show=False
        )
        st.plotly_chart(fig2, use_container_width=True)

        # 月度收益率热力图
        fig3 = plotter.plot_monthly_returns(
            result.returns,
            title="月度收益率热力图",
            show=False
        )
        st.plotly_chart(fig3, use_container_width=True)

        # 滚动指标
        fig4 = plotter.plot_rolling_metrics(
            result.equity_curve,
            title="滚动指标",
            show=False
        )
        st.plotly_chart(fig4, use_container_width=True)

        # 交易记录
        if not result.trades.empty:
            st.markdown('<h3 class="section-header">💼 交易记录</h3>', unsafe_allow_html=True)

            # 格式化交易记录
            trades_display = result.trades.copy()
            if not trades_display.empty:
                trades_display = trades_display.reset_index()
                trades_display['date'] = trades_display['date'].dt.strftime('%Y-%m-%d')
                trades_display['value'] = trades_display['value'].apply(lambda x: f"¥{x:,.2f}")
                trades_display['commission'] = trades_display['commission'].apply(lambda x: f"¥{x:,.2f}")
                if 'pnl' in trades_display.columns:
                    trades_display['pnl'] = trades_display['pnl'].apply(lambda x: f"¥{x:,.2f}")

                st.dataframe(trades_display, use_container_width=True)

        # 持仓记录
        if not result.positions.empty:
            st.markdown('<h3 class="section-header">📦 持仓记录</h3>', unsafe_allow_html=True)

            # 格式化持仓记录
            positions_display = result.positions.copy()
            if not positions_display.empty:
                positions_display = positions_display.reset_index()
                positions_display['date'] = positions_display['date'].dt.strftime('%Y-%m-%d')
                positions_display['value'] = positions_display['value'].apply(lambda x: f"¥{x:,.2f}")

                st.dataframe(positions_display, use_container_width=True)

        # 下载按钮
        st.markdown('<h3 class="section-header">💾 数据导出</h3>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            # 导出权益曲线
            equity_csv = result.equity_curve.reset_index().to_csv(index=False)
            st.download_button(
                label="📥 下载权益曲线",
                data=equity_csv,
                file_name="equity_curve.csv",
                mime="text/csv"
            )

        with col2:
            # 导出交易记录
            if not result.trades.empty:
                trades_csv = result.trades.reset_index().to_csv(index=False)
                st.download_button(
                    label="📥 下载交易记录",
                    data=trades_csv,
                    file_name="trades.csv",
                    mime="text/csv"
                )

        with col3:
            # 导出性能指标
            import json
            metrics_json = json.dumps(result.performance_metrics, indent=2, ensure_ascii=False)
            st.download_button(
                label="📥 下载性能指标",
                data=metrics_json,
                file_name="performance_metrics.json",
                mime="application/json"
            )

        progress_bar.progress(1.0)
        st.success("✅ 回测完成!")

    except Exception as e:
        st.error(f"❌ 运行过程中出错: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

else:
    # 显示欢迎信息和示例
    st.markdown("""
    ## 🎯 欢迎使用量化选股软件

    这是一个基于Python的量化选股平台，支持多种策略和因子分析。

    ### 🚀 快速开始

    1. **配置参数**：在左侧边栏设置股票代码、日期范围和策略参数
    2. **选择因子**：选择要计算的技术指标和基本面因子
    3. **开始回测**：点击"开始回测"按钮运行策略

    ### 📊 支持的功能

    - **多种策略**：动量策略、双动量策略、均值回归策略、多因子策略
    - **丰富因子**：100+技术指标和基本面因子
    - **完整回测**：支持手续费、滑点、仓位管理等
    - **可视化分析**：权益曲线、收益率分布、热力图等
    - **性能评估**：夏普比率、最大回撤、Alpha/Beta等指标

    ### ⚙️ 默认配置

    - **股票代码**：600519(贵州茅台), 000858(五粮液), 000333(美的集团), 601318(中国平安), 600036(招商银行)
    - **日期范围**：最近一年
    - **策略类型**：动量策略
    - **初始资金**：100万元
    - **数据源**：Akshare (支持A股/港股/美股)

    ### 📈 开始使用

    请在左侧边栏配置参数，然后点击"开始回测"按钮。
    """)

    # 显示示例图表
    st.markdown("### 📊 示例图表")

    # 创建示例数据
    dates = pd.date_range('2023-01-01', periods=252, freq='B')
    np.random.seed(42)

    # 示例权益曲线
    returns = np.random.randn(len(dates)) * 0.01 + 0.0005
    equity_curve = pd.Series(1000000 * (1 + returns).cumprod(), index=dates)

    # 示例基准曲线
    benchmark_returns = np.random.randn(len(dates)) * 0.008 + 0.0002
    benchmark_curve = pd.Series(1000000 * (1 + benchmark_returns).cumprod(), index=dates)

    # 创建示例图表
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=equity_curve.values,
        mode='lines', name='策略净值',
        line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=benchmark_curve.values,
        mode='lines', name='基准净值',
        line=dict(color='red', width=2, dash='dash')
    ))

    fig.update_layout(
        title="示例权益曲线",
        xaxis_title="日期",
        yaxis_title="净值",
        template="plotly_white",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

# 页脚
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>📈 量化选股软件 v1.0 | 基于Python的量化投资平台</p>
    <p>⚠️ 风险提示：本软件仅供学习和研究使用，不构成投资建议</p>
</div>
""", unsafe_allow_html=True)