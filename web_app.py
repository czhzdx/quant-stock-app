"""量化选股软件Web界面 - 科技风格"""
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
    page_title="QuantStock Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 科技风格CSS
st.markdown("""
<style>
    /* 导入字体 */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');

    /* 全局变量 */
    :root {
        --bg-primary: #0a0e17;
        --bg-secondary: #0d1321;
        --bg-card: #111827;
        --accent-cyan: #00f5ff;
        --accent-purple: #b24bf3;
        --accent-blue: #3b82f6;
        --accent-green: #10b981;
        --accent-red: #ef4444;
        --accent-orange: #f59e0b;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --border-color: #1e293b;
        --glow-cyan: 0 0 20px rgba(0, 245, 255, 0.3);
        --glow-purple: 0 0 20px rgba(178, 75, 243, 0.3);
    }

    /* 隐藏默认元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* 主背景 */
    .stApp {
        background: linear-gradient(135deg, #0a0e17 0%, #0d1321 50%, #111827 100%);
        background-attachment: fixed;
    }

    /* 侧边栏样式 */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1321 0%, #111827 100%) !important;
        border-right: 1px solid #1e293b !important;
    }

    section[data-testid="stSidebar"] .element-container {
        margin-bottom: 0.5rem;
    }

    /* 标题样式 */
    .main-title {
        font-family: 'Orbitron', monospace;
        font-size: 2.8rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #00f5ff, #b24bf3, #00f5ff);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient 3s linear infinite;
        text-shadow: 0 0 30px rgba(0, 245, 255, 0.5);
        margin-bottom: 0.5rem;
        letter-spacing: 4px;
    }

    .sub-title {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1rem;
        color: #64748b;
        text-align: center;
        letter-spacing: 8px;
        text-transform: uppercase;
        margin-bottom: 2rem;
    }

    @keyframes gradient {
        0% { background-position: 0% center; }
        50% { background-position: 100% center; }
        100% { background-position: 0% center; }
    }

    /* 区块标题 */
    .section-header {
        font-family: 'Orbitron', monospace;
        font-size: 1.3rem;
        font-weight: 600;
        color: #00f5ff;
        margin: 1.5rem 0 1rem 0;
        padding: 0.8rem 1rem;
        background: linear-gradient(90deg, rgba(0, 245, 255, 0.1), transparent);
        border-left: 3px solid #00f5ff;
        border-radius: 0 8px 8px 0;
        letter-spacing: 2px;
    }

    /* 数据卡片 */
    .metric-card {
        background: linear-gradient(145deg, #111827, #0f172a);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 1.2rem;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00f5ff, transparent);
    }

    .metric-card:hover {
        border-color: #00f5ff;
        box-shadow: 0 0 20px rgba(0, 245, 255, 0.15);
        transform: translateY(-2px);
    }

    .metric-label {
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .metric-value {
        font-family: 'Orbitron', monospace;
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 0.3rem;
    }

    .metric-value.positive { color: #10b981; }
    .metric-value.negative { color: #ef4444; }
    .metric-value.neutral { color: #f59e0b; }

    .metric-change {
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.9rem;
        margin-top: 0.3rem;
    }

    /* 评级样式 */
    .rating-badge {
        display: inline-block;
        padding: 0.5rem 1.2rem;
        border-radius: 20px;
        font-family: 'Orbitron', monospace;
        font-weight: 600;
        font-size: 1rem;
        letter-spacing: 1px;
    }

    .rating-strong-buy {
        background: linear-gradient(135deg, rgba(236, 72, 153, 0.2), rgba(236, 72, 153, 0.1));
        color: #ec4899;
        border: 1px solid #ec4899;
        box-shadow: 0 0 15px rgba(236, 72, 153, 0.3);
    }

    .rating-buy {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(16, 185, 129, 0.1));
        color: #10b981;
        border: 1px solid #10b981;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.3);
    }

    .rating-neutral {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.2), rgba(245, 158, 11, 0.1));
        color: #f59e0b;
        border: 1px solid #f59e0b;
        box-shadow: 0 0 15px rgba(245, 158, 11, 0.3);
    }

    .rating-sell {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(239, 68, 68, 0.1));
        color: #ef4444;
        border: 1px solid #ef4444;
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.3);
    }

    /* 标签页样式 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Orbitron', monospace;
        font-size: 0.9rem;
        font-weight: 500;
        padding: 12px 24px;
        background: linear-gradient(145deg, #111827, #0f172a);
        border: 1px solid #1e293b;
        border-radius: 8px 8px 0 0;
        color: #64748b;
        transition: all 0.3s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #00f5ff;
        border-color: #00f5ff;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(145deg, #0f172a, #111827) !important;
        color: #00f5ff !important;
        border-color: #00f5ff !important;
        border-bottom: 2px solid #00f5ff !important;
        box-shadow: 0 -5px 20px rgba(0, 245, 255, 0.1);
    }

    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }

    /* 按钮样式 */
    .stButton button {
        font-family: 'Orbitron', monospace;
        font-weight: 600;
        letter-spacing: 1px;
        background: linear-gradient(135deg, #00f5ff, #b24bf3) !important;
        color: #0a0e17 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        transition: all 0.3s ease !important;
    }

    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 30px rgba(0, 245, 255, 0.4) !important;
    }

    .stButton button:active {
        transform: translateY(0);
    }

    /* 输入框样式 */
    .stTextInput input, .stTextArea textarea, .stSelectbox div, .stMultiSelect div {
        background: #111827 !important;
        border: 1px solid #1e293b !important;
        border-radius: 8px !important;
        color: #f8fafc !important;
    }

    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #00f5ff !important;
        box-shadow: 0 0 10px rgba(0, 245, 255, 0.2) !important;
    }

    .stTextInput label, .stTextArea label, .stSelectbox label, .stMultiSelect label {
        font-family: 'Rajdhani', sans-serif;
        font-weight: 500;
        color: #94a3b8 !important;
    }

    /* 滑块样式 */
    .stSlider div[role="slider"] {
        background: linear-gradient(90deg, #00f5ff, #b24bf3);
    }

    /* 复选框样式 */
    .stCheckbox label {
        font-family: 'Rajdhani', sans-serif;
        color: #94a3b8;
    }

    /* 日期输入 */
    .stDateInput input {
        background: #111827 !important;
        border: 1px solid #1e293b !important;
    }

    /* 数据框样式 */
    .stDataFrame {
        border: 1px solid #1e293b;
        border-radius: 12px;
        overflow: hidden;
    }

    .stDataFrame table {
        background: #111827 !important;
    }

    .stDataFrame thead th {
        background: #0f172a !important;
        color: #00f5ff !important;
        font-family: 'Orbitron', monospace;
        font-weight: 600;
        letter-spacing: 1px;
    }

    .stDataFrame tbody td {
        color: #f8fafc !important;
        font-family: 'Rajdhani', sans-serif;
    }

    .stDataFrame tbody tr:hover {
        background: rgba(0, 245, 255, 0.05) !important;
    }

    /* 进度条样式 */
    .stProgress > div > div {
        background: #1e293b;
    }

    .stProgress > div > div > div {
        background: linear-gradient(90deg, #00f5ff, #b24bf3);
    }

    /* 成功/警告/错误消息 */
    .stSuccess {
        background: rgba(16, 185, 129, 0.1) !important;
        border: 1px solid #10b981 !important;
        color: #10b981 !important;
        border-radius: 8px !important;
    }

    .stWarning {
        background: rgba(245, 158, 11, 0.1) !important;
        border: 1px solid #f59e0b !important;
        color: #f59e0b !important;
        border-radius: 8px !important;
    }

    .stError {
        background: rgba(239, 68, 68, 0.1) !important;
        border: 1px solid #ef4444 !important;
        color: #ef4444 !important;
        border-radius: 8px !important;
    }

    .stInfo {
        background: rgba(59, 130, 246, 0.1) !important;
        border: 1px solid #3b82f6 !important;
        color: #3b82f6 !important;
        border-radius: 8px !important;
    }

    /* 侧边栏标题 */
    section[data-testid="stSidebar"] h2 {
        font-family: 'Orbitron', monospace;
        color: #00f5ff;
        letter-spacing: 2px;
    }

    section[data-testid="stSidebar"] h3 {
        font-family: 'Orbitron', monospace;
        color: #b24bf3;
        font-size: 0.9rem;
        letter-spacing: 1px;
        margin-top: 1.5rem;
    }

    section[data-testid="stSidebar"] hr {
        border-color: #1e293b;
    }

    /* Expander样式 */
    .streamlit-expanderHeader {
        font-family: 'Rajdhani', sans-serif;
        background: linear-gradient(145deg, #111827, #0f172a);
        border: 1px solid #1e293b;
        border-radius: 8px !important;
    }

    /* 下载按钮 */
    .stDownloadButton button {
        background: transparent !important;
        border: 1px solid #00f5ff !important;
        color: #00f5ff !important;
    }

    .stDownloadButton button:hover {
        background: rgba(0, 245, 255, 0.1) !important;
    }

    /* Plotly图表背景 */
    .js-plotly-plot .plotly .bg {
        fill: #111827 !important;
    }

    /* 页脚 */
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 2rem;
        border-top: 1px solid #1e293b;
    }

    .footer p {
        font-family: 'Rajdhani', sans-serif;
        color: #64748b;
        margin: 0.3rem 0;
    }

    .footer .version {
        font-family: 'Orbitron', monospace;
        color: #00f5ff;
        font-size: 0.85rem;
        letter-spacing: 2px;
    }

    /* 滚动条样式 */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #0a0e17;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #00f5ff, #b24bf3);
        border-radius: 4px;
    }

    /* 加载动画 */
    .stSpinner > div {
        border-color: #00f5ff transparent transparent transparent !important;
    }

    /* 分隔线 */
    hr {
        border-color: #1e293b !important;
    }

    /* 分析结果卡片 */
    .analysis-card {
        background: linear-gradient(145deg, #111827, #0f172a);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    .analysis-card h4 {
        font-family: 'Orbitron', monospace;
        color: #b24bf3;
        font-size: 1rem;
        margin-bottom: 0.8rem;
    }

    .analysis-card p {
        font-family: 'Rajdhani', sans-serif;
        color: #94a3b8;
        line-height: 1.6;
    }

    /* 网格背景装饰 */
    .grid-bg {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-image:
            linear-gradient(rgba(0, 245, 255, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 245, 255, 0.03) 1px, transparent 1px);
        background-size: 50px 50px;
        pointer-events: none;
        z-index: -1;
    }

    /* 数字动画效果 */
    .number-glow {
        text-shadow: 0 0 10px currentColor;
    }
</style>

<div class="grid-bg"></div>
""", unsafe_allow_html=True)

# 标题
st.markdown('<h1 class="main-title">QUANTSTOCK PRO</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">AI-Powered Quantitative Trading Platform</p>', unsafe_allow_html=True)

# 创建标签页
tab1, tab2 = st.tabs(["⚡ STRATEGY BACKTEST", "🤖 AI ANALYSIS"])


# ========== 标签页1: 策略回测 ==========
with tab1:
    # 侧边栏配置
    with st.sidebar:
        st.markdown("## ⚙️ CONFIG")

        # 数据获取配置
        st.markdown("### 📊 DATA")
        st.markdown("<small style='color:#64748b'>A股: 600519 | 港股: 00700 | 美股: AAPL</small>", unsafe_allow_html=True)

        symbols_input = st.text_area(
            "股票代码",
            value="600519\n000858\n000333\n601318\n600036",
            height=100,
            key="backtest_symbols",
            label_visibility="collapsed"
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
            start_date = st.date_input("开始", value=datetime.now() - timedelta(days=365), key="backtest_start")
        with col2:
            end_date = st.date_input("结束", value=datetime.now(), key="backtest_end")

        interval = st.selectbox("频率", options=["1d", "1wk", "1mo"], index=0, key="backtest_interval")

        # 策略配置
        st.markdown("### 🎯 STRATEGY")
        strategy_type = st.selectbox(
            "策略",
            options=["动量策略", "双动量策略", "均值回归策略", "多因子策略"],
            index=0, key="backtest_strategy", label_visibility="collapsed"
        )

        if strategy_type == "动量策略":
            lookback = st.slider("回顾期", 5, 250, 20, key="bt_lookback")
            holding = st.slider("持有期", 1, 60, 10, key="bt_holding")
            top_n = st.slider("Top N", 1, 20, 5, key="bt_top_n")
        elif strategy_type == "双动量策略":
            abs_lookback = st.slider("绝对动量期", 5, 100, 20, key="bt_abs_lookback")
            rel_lookback = st.slider("相对动量期", 20, 250, 60, key="bt_rel_lookback")
            top_n = st.slider("Top N", 1, 20, 5, key="bt_top_n2")
        elif strategy_type == "均值回归策略":
            lookback = st.slider("回顾期", 5, 250, 20, key="bt_lookback2")
            zscore_threshold = st.slider("Z分数阈值", 1.0, 3.0, 2.0, 0.1, key="bt_zscore")
        else:
            top_n = st.slider("Top N", 1, 20, 5, key="bt_top_n3")
            rebalance_freq = st.selectbox("再平衡", ["每日", "每周", "每月"], index=2, key="bt_rebalance")

        # 回测配置
        st.markdown("### 💰 CAPITAL")
        initial_capital = st.number_input("初始资金", min_value=10000, max_value=100000000, value=1000000, step=10000, key="bt_capital")

        col1, col2 = st.columns(2)
        with col1:
            commission_rate = st.number_input("手续费%", min_value=0.0, max_value=1.0, value=0.03, step=0.01, key="bt_commission") / 100
        with col2:
            slippage_rate = st.number_input("滑点%", min_value=0.0, max_value=1.0, value=0.01, step=0.01, key="bt_slippage") / 100

        # 因子配置
        st.markdown("### 🔢 FACTORS")
        factor_calculator = FactorCalculator()
        technical_factors = factor_calculator.list_factors(FactorType.TECHNICAL)
        default_technical = ["ma_20", "macd"]
        rsi_factors = [f for f in technical_factors if f.startswith("rsi_")]
        if rsi_factors:
            default_technical.append(rsi_factors[0])
        selected_technical = st.multiselect("技术指标", options=technical_factors, default=default_technical, key="bt_technical", label_visibility="collapsed")

        # 执行按钮
        st.markdown("---")
        run_backtest = st.button("🚀 EXECUTE BACKTEST", type="primary", use_container_width=True, key="btn_backtest")

    # 回测主逻辑
    if run_backtest and symbols:
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()

            # 获取数据
            status_text.markdown("<p style='color:#00f5ff'>📥 Fetching market data...</p>", unsafe_allow_html=True)
            fetcher = DataFetcher(cache_enabled=True)

            stock_data = {}
            for i, symbol in enumerate(symbols):
                try:
                    data = fetcher.get_stock_data(
                        symbol=symbol,
                        start_date=start_date.strftime("%Y-%m-%d"),
                        end_date=end_date.strftime("%Y-%m-%d"),
                        interval=interval, adjust=True
                    )
                    if data is not None and not data.empty:
                        stock_data[symbol] = data
                        st.success(f"✓ {symbol} ({len(data)} records)")
                except Exception as e:
                    st.error(f"✗ {symbol}: {str(e)}")
                progress_bar.progress((i + 1) / len(symbols) * 0.3)

            if not stock_data:
                st.error("No data retrieved")
                st.stop()

            # 计算因子
            status_text.markdown("<p style='color:#00f5ff'>🔢 Computing factors...</p>", unsafe_allow_html=True)
            calculator = FactorCalculator()
            factor_data = {}
            selected_factors = selected_technical

            for i, (symbol, data) in enumerate(stock_data.items()):
                if selected_factors:
                    try:
                        factors = calculator.calculate_multiple_factors(data, selected_factors)
                        factor_data[symbol] = factors
                    except Exception as e:
                        st.error(f"✗ {symbol}: {str(e)}")
                progress_bar.progress(0.3 + (i + 1) / len(stock_data) * 0.2)

            # 创建策略
            status_text.markdown("<p style='color:#00f5ff'>🎯 Building strategy...</p>", unsafe_allow_html=True)
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
                    top_n=top_n, rebalance_freq=freq_map[rebalance_freq]
                )

            progress_bar.progress(0.6)

            # 准备回测数据
            if strategy_type in ["动量策略", "双动量策略"]:
                backtest_data = pd.DataFrame({s: d['Close'] for s, d in stock_data.items()})
                mode = BacktestMode.MULTI_STOCK
            else:
                first_symbol = list(stock_data.keys())[0]
                backtest_data = stock_data[first_symbol]
                mode = BacktestMode.SINGLE_STOCK

            progress_bar.progress(0.7)

            # 运行回测
            status_text.markdown("<p style='color:#00f5ff'>⚡ Running backtest...</p>", unsafe_allow_html=True)
            backtester = Backtester(strategy=strategy, initial_capital=initial_capital, commission_rate=commission_rate, slippage_rate=slippage_rate)
            result = backtester.run(data=backtest_data, start_date=start_date.strftime("%Y-%m-%d"), end_date=end_date.strftime("%Y-%m-%d"), mode=mode)

            progress_bar.progress(0.9)
            status_text.markdown("<p style='color:#00f5ff'>📈 Generating report...</p>", unsafe_allow_html=True)
            progress_bar.empty()
            status_text.empty()

            # 显示结果
            st.markdown('<h2 class="section-header">📊 PERFORMANCE SUMMARY</h2>', unsafe_allow_html=True)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                total_return = result.performance_metrics.get('总收益率%', 0)
                return_class = "positive" if total_return > 0 else "negative"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">TOTAL RETURN</div>
                    <div class="metric-value {return_class} number-glow">{total_return:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">ANNUAL RETURN</div>
                    <div class="metric-value">{result.performance_metrics.get('年化收益率%', 0):.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                sharpe = result.performance_metrics.get('夏普比率', 0)
                sharpe_class = "positive" if sharpe > 1 else "neutral" if sharpe > 0 else "negative"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">SHARPE RATIO</div>
                    <div class="metric-value {sharpe_class}">{sharpe:.3f}</div>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                max_dd = result.performance_metrics.get('最大回撤%', 0)
                dd_class = "negative" if max_dd < -20 else "neutral" if max_dd < -10 else "positive"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">MAX DRAWDOWN</div>
                    <div class="metric-value {dd_class}">{max_dd:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)

            # 详细指标
            st.markdown('<h3 class="section-header">📈 DETAILED METRICS</h3>', unsafe_allow_html=True)

            metrics_data = [
                ["INITIAL CAPITAL", f"¥{result.performance_metrics.get('初始资金', 0):,.0f}"],
                ["FINAL EQUITY", f"¥{result.performance_metrics.get('最终权益', 0):,.0f}"],
                ["ANNUAL VOLATILITY", f"{result.performance_metrics.get('年化波动率%', 0):.2f}%"],
                ["SORTINO RATIO", f"{result.performance_metrics.get('索提诺比率', 0):.3f}"],
                ["CALMAR RATIO", f"{result.performance_metrics.get('卡尔玛比率', 0):.3f}"],
                ["WIN RATE", f"{result.performance_metrics.get('胜率%', 0):.2f}%"],
                ["PROFIT/LOSS RATIO", f"{result.performance_metrics.get('盈亏比', 0):.2f}"],
                ["TOTAL TRADES", f"{result.performance_metrics.get('总交易次数', 0)}"],
                ["ALPHA", f"{result.performance_metrics.get('Alpha', 0):.3f}"],
                ["BETA", f"{result.performance_metrics.get('Beta', 0):.3f}"],
            ]

            metrics_df = pd.DataFrame(metrics_data, columns=["METRIC", "VALUE"])
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

            # 图表
            st.markdown('<h3 class="section-header">📊 VISUALIZATION</h3>', unsafe_allow_html=True)
            plotter = Plotter()

            fig1 = plotter.plot_equity_curve(result.equity_curve, result.benchmark_returns, title="EQUITY CURVE", show=False)
            fig1.update_layout(paper_bgcolor='#111827', plot_bgcolor='#111827', font_color='#94a3b8')
            st.plotly_chart(fig1, use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                fig2 = plotter.plot_returns_distribution(result.returns, title="RETURNS DISTRIBUTION", show=False)
                fig2.update_layout(paper_bgcolor='#111827', plot_bgcolor='#111827', font_color='#94a3b8')
                st.plotly_chart(fig2, use_container_width=True)
            with col2:
                fig3 = plotter.plot_monthly_returns(result.returns, title="MONTHLY RETURNS", show=False)
                fig3.update_layout(paper_bgcolor='#111827', plot_bgcolor='#111827', font_color='#94a3b8')
                st.plotly_chart(fig3, use_container_width=True)

            st.success("✓ BACKTEST COMPLETE")

        except Exception as e:
            st.error(f"✗ Error: {str(e)}")
            import traceback
            with st.expander("Details"):
                st.code(traceback.format_exc())
    else:
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
            <h3 style="color: #64748b; font-family: 'Orbitron', monospace;">← Configure parameters in sidebar</h3>
            <p style="color: #475569;">Click "EXECUTE BACKTEST" to start</p>
        </div>
        """, unsafe_allow_html=True)


# ========== 标签页2: AI智能分析 ==========
with tab2:
    with st.sidebar:
        st.markdown("## 🤖 AI CONFIG")

        st.markdown("### 📊 STOCKS")
        analyze_symbols = st.text_input("股票代码 (逗号分隔)", value="600519,000858,000333", key="analyze_symbols", label_visibility="collapsed")
        analyze_stocks = [s.strip() for s in analyze_symbols.split(',') if s.strip()]

        st.markdown("### 📅 DATE RANGE")
        col1, col2 = st.columns(2)
        with col1:
            analyze_start = st.date_input("开始", value=datetime.now() - timedelta(days=180), key="analyze_start")
        with col2:
            analyze_end = st.date_input("结束", value=datetime.now(), key="analyze_end")

        st.markdown("### 🔧 OPTIONS")
        enable_search = st.checkbox("搜索增强", value=True, key="analyze_search")
        enable_chip = st.checkbox("筹码数据", value=True, key="analyze_chip")
        enable_realtime = st.checkbox("实时行情", value=True, key="analyze_realtime")

        st.markdown("### 📬 NOTIFICATIONS")
        enable_notify = st.checkbox("发送通知", value=False, key="analyze_notify")
        notify_channels = st.multiselect("渠道", options=["email", "wechat"], default=["email"], key="analyze_channels", label_visibility="collapsed")

        st.markdown("---")
        run_analyze = st.button("🤖 START AI ANALYSIS", type="primary", use_container_width=True, key="btn_analyze")

    if run_analyze and analyze_stocks:
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()

            # 初始化组件
            status_text.markdown("<p style='color:#00f5ff'>🔧 Initializing AI engine...</p>", unsafe_allow_html=True)
            data_fetcher = EnhancedDataFetcher()
            search_manager = SearchAPIManager() if enable_search else None
            llm_analyzer = LLMAnalyzer()
            report_generator = ReportGenerator()
            notifier = Notifier() if enable_notify else None

            progress_bar.progress(0.1)
            analyses_results = []

            for i, symbol in enumerate(analyze_stocks):
                status_text.markdown(f"<p style='color:#00f5ff'>📊 Analyzing {symbol}...</p>", unsafe_allow_html=True)

                with st.spinner(f"Fetching {symbol} data..."):
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
                    change_class = "positive" if rt.change_pct > 0 else "negative"

                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span style="font-family: 'Orbitron', monospace; font-size: 1.5rem; color: #f8fafc;">{rt.name}</span>
                                <span style="color: #64748b; margin-left: 1rem;">{symbol}</span>
                            </div>
                            <div style="text-align: right;">
                                <div class="metric-value {change_class}">¥{rt.price:.2f}</div>
                                <div class="metric-change {change_class}">{rt.change_pct:+.2f}%</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("VOLUME", f"{rt.volume:,}")
                    with col2:
                        st.metric("TURNOVER", f"{rt.turnover_rate:.2f}%")
                    with col3:
                        st.metric("AMOUNT", f"¥{rt.amount:,.0f}")
                    stock_name = rt.name
                else:
                    st.markdown(f"<div class='metric-card'><span class='metric-value'>{symbol}</span></div>", unsafe_allow_html=True)
                    stock_name = symbol

                # 搜索增强
                search_info = None
                if search_manager and enable_search:
                    status_text.markdown(f"<p style='color:#00f5ff'>🔍 Searching {symbol} info...</p>", unsafe_allow_html=True)
                    with st.spinner("Searching..."):
                        search_info = asyncio.run(search_manager.search_stock_info(symbol=symbol, stock_name=stock_name))

                    if search_info and search_info.news:
                        with st.expander("📰 RECENT NEWS", expanded=False):
                            for news in search_info.news[:5]:
                                st.markdown(f"- **{news.title}**")

                progress_bar.progress(0.3 + (i / len(analyze_stocks)) * 0.3)

                # LLM分析
                status_text.markdown(f"<p style='color:#00f5ff'>🤖 AI analyzing {symbol}...</p>", unsafe_allow_html=True)
                with st.spinner("Running AI analysis..."):
                    analysis = asyncio.run(llm_analyzer.analyze_stock(stock_data=stock_data, search_info=search_info, stock_name=stock_name))

                progress_bar.progress(0.6 + (i / len(analyze_stocks)) * 0.3)

                # 显示分析结果
                st.markdown("---")

                # 评级
                rating_class = {
                    "强烈推荐": "rating-strong-buy",
                    "推荐": "rating-buy",
                    "中性": "rating-neutral",
                    "不推荐": "rating-sell",
                    "强烈不推荐": "rating-sell"
                }.get(analysis.rating, "rating-neutral")

                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.markdown(f"**RATING**: <span class='rating-badge {rating_class}'>{analysis.rating}</span>", unsafe_allow_html=True)
                with col2:
                    if analysis.target_price:
                        st.metric("TARGET", f"¥{analysis.target_price:.2f}")
                with col3:
                    if analysis.stop_loss:
                        st.metric("STOP LOSS", f"¥{analysis.stop_loss:.2f}")

                # 详细分析标签
                tab_a, tab_b, tab_c, tab_d = st.tabs(["📊 FUNDAMENTAL", "📈 TECHNICAL", "🎯 CHIP ANALYSIS", "💡 ADVICE"])

                with tab_a:
                    if analysis.fundamental_analysis:
                        st.markdown(f"<div class='analysis-card'>{analysis.fundamental_analysis}</div>", unsafe_allow_html=True)
                    else:
                        st.info("No fundamental analysis available")

                with tab_b:
                    if analysis.technical_analysis:
                        st.markdown(f"<div class='analysis-card'>{analysis.technical_analysis}</div>", unsafe_allow_html=True)
                    else:
                        st.info("No technical analysis available")

                with tab_c:
                    if analysis.chip_analysis:
                        st.markdown(f"<div class='analysis-card'>{analysis.chip_analysis}</div>", unsafe_allow_html=True)
                    elif stock_data.chip_distribution:
                        chip = stock_data.chip_distribution
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("PROFIT RATIO", f"{chip.profit_ratio:.2f}%")
                        with col2:
                            st.metric("AVG COST", f"¥{chip.avg_cost:.2f}")
                    else:
                        st.info("No chip data available")

                with tab_d:
                    if analysis.operation_advice:
                        st.markdown(f"<div class='analysis-card'>{analysis.operation_advice}</div>", unsafe_allow_html=True)
                    else:
                        st.info("No advice available")

                if analysis.risks:
                    st.warning("⚠️ **RISKS**: " + ", ".join(analysis.risks))

                analyses_results.append(analysis)

                if notifier and enable_notify:
                    summary = analysis.technical_analysis[:500] if analysis.technical_analysis else ""
                    asyncio.run(notifier.send_analysis_report(symbol=symbol, stock_name=stock_name, rating=analysis.rating, summary=summary, channels=notify_channels))

            progress_bar.progress(1.0)
            progress_bar.empty()
            status_text.empty()

            # 汇总
            if len(analyses_results) > 1:
                st.markdown("---")
                st.markdown('<h2 class="section-header">📊 ANALYSIS SUMMARY</h2>', unsafe_allow_html=True)

                summary_data = []
                for a in analyses_results:
                    summary_data.append({
                        "STOCK": f"{a.stock_name} ({a.symbol})",
                        "RATING": a.rating,
                        "TARGET": f"¥{a.target_price:.2f}" if a.target_price else "-",
                        "STOP LOSS": f"¥{a.stop_loss:.2f}" if a.stop_loss else "-",
                        "MODEL": a.model_used
                    })

                st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

                report_files = report_generator.generate_summary_report(analyses_results)
                st.success(f"✓ Report generated: {list(report_files.values())}")

            st.success("✓ AI ANALYSIS COMPLETE")

        except Exception as e:
            st.error(f"✗ Error: {str(e)}")
            import traceback
            with st.expander("Details"):
                st.code(traceback.format_exc())
    else:
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
            <h3 style="color: #64748b; font-family: 'Orbitron', monospace;">← Configure AI analysis in sidebar</h3>
            <p style="color: #475569;">Enter stock codes and click "START AI ANALYSIS"</p>
        </div>

        <div style="max-width: 800px; margin: 2rem auto;">
            <h3 class="section-header">🤖 AI-POWERED ANALYSIS</h3>
            <div class="analysis-card">
                <p>Advanced stock analysis powered by Large Language Models (LLM):</p>
                <ul style="color: #94a3b8; line-height: 2;">
                    <li><strong style="color:#00f5ff;">Fundamental Analysis</strong> - Business overview, industry position, key risks</li>
                    <li><strong style="color:#00f5ff;">Technical Analysis</strong> - Trend analysis, support/resistance, volume patterns</li>
                    <li><strong style="color:#00f5ff;">Chip Analysis</strong> - Distribution, profit ratio, institutional flow</li>
                    <li><strong style="color:#00f5ff;">News Analysis</strong> - Market sentiment, announcement impact</li>
                    <li><strong style="color:#00f5ff;">Investment Rating</strong> - Buy/Sell/Hold with target prices</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)


# 页脚
st.markdown("""
<div class="footer">
    <p class="version">QUANTSTOCK PRO v2.0</p>
    <p>Strategy Backtest • AI Analysis • Real-time Data</p>
    <p style="font-size: 0.8rem;">⚠️ For educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)
