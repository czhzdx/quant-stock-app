"""量化选股软件Web界面 - 科技风格"""
import sys
import os
import asyncio
from pathlib import Path

root_dir = Path(__file__).parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from src.data.data_fetcher import DataFetcher
from src.factors.factor_calculator import FactorCalculator, FactorType
from src.strategies.momentum import MomentumStrategy, DualMomentumStrategy, MeanReversionStrategy, MultiFactorStrategy
from src.backtest.backtester import Backtester, BacktestMode
from src.visualization.plotter import Plotter

from src.analysis.data_enhanced import EnhancedDataFetcher
from src.analysis.search_apis import SearchAPIManager
from src.analysis.llm_analyzer import LLMAnalyzer
from src.analysis.report_generator import ReportGenerator
from src.analysis.notifier import Notifier

st.set_page_config(
    page_title="QuantStock Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 科技风格CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');

    :root {
        --bg-primary: #0a0e17;
        --bg-secondary: #0d1321;
        --bg-card: #111827;
        --accent-cyan: #00f5ff;
        --accent-purple: #b24bf3;
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --border-color: #1e293b;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stApp {
        background: linear-gradient(135deg, #0a0e17 0%, #0d1321 50%, #111827 100%);
        background-attachment: fixed;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1321 0%, #111827 100%) !important;
        border-right: 1px solid #1e293b !important;
    }

    .main-title {
        font-family: 'Orbitron', monospace;
        font-size: 2.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #00f5ff, #b24bf3, #00f5ff);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradient 3s linear infinite;
        margin-bottom: 0.3rem;
        letter-spacing: 4px;
    }

    .sub-title {
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.9rem;
        color: #64748b;
        text-align: center;
        letter-spacing: 6px;
        text-transform: uppercase;
        margin-bottom: 1.5rem;
    }

    @keyframes gradient {
        0% { background-position: 0% center; }
        50% { background-position: 100% center; }
        100% { background-position: 0% center; }
    }

    .section-header {
        font-family: 'Orbitron', monospace;
        font-size: 1.2rem;
        font-weight: 600;
        color: #00f5ff;
        margin: 1.2rem 0 0.8rem 0;
        padding: 0.6rem 1rem;
        background: linear-gradient(90deg, rgba(0, 245, 255, 0.1), transparent);
        border-left: 3px solid #00f5ff;
        border-radius: 0 8px 8px 0;
        letter-spacing: 2px;
    }

    .section-header.purple {
        color: #b24bf3;
        border-left-color: #b24bf3;
        background: linear-gradient(90deg, rgba(178, 75, 243, 0.1), transparent);
    }

    .metric-card {
        background: linear-gradient(145deg, #111827, #0f172a);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 1rem;
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00f5ff, transparent);
    }

    .metric-label {
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
    }

    .metric-value {
        font-family: 'Orbitron', monospace;
        font-size: 1.5rem;
        font-weight: 700;
        color: #f8fafc;
    }

    .metric-value.positive { color: #10b981; }
    .metric-value.negative { color: #ef4444; }

    .rating-badge {
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-family: 'Orbitron', monospace;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .rating-strong-buy { background: rgba(236, 72, 153, 0.2); color: #ec4899; border: 1px solid #ec4899; }
    .rating-buy { background: rgba(16, 185, 129, 0.2); color: #10b981; border: 1px solid #10b981; }
    .rating-neutral { background: rgba(245, 158, 11, 0.2); color: #f59e0b; border: 1px solid #f59e0b; }
    .rating-sell { background: rgba(239, 68, 68, 0.2); color: #ef4444; border: 1px solid #ef4444; }

    /* 标签页样式 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Orbitron', monospace;
        font-size: 0.95rem;
        font-weight: 600;
        padding: 12px 32px;
        background: linear-gradient(145deg, #111827, #0f172a);
        border: 1px solid #1e293b;
        border-radius: 10px;
        color: #64748b;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #00f5ff;
        border-color: #00f5ff;
    }
    .stTabs [aria-selected="true"] {
        color: #00f5ff !important;
        border-color: #00f5ff !important;
        box-shadow: 0 0 25px rgba(0, 245, 255, 0.2);
    }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }

    .stButton button {
        font-family: 'Orbitron', monospace;
        font-weight: 600;
        background: linear-gradient(135deg, #00f5ff, #b24bf3) !important;
        color: #0a0e17 !important;
        border: none !important;
        border-radius: 8px !important;
    }
    .stButton button:hover {
        box-shadow: 0 0 30px rgba(0, 245, 255, 0.4) !important;
    }

    .stTextInput input, .stTextArea textarea, .stSelectbox div, .stMultiSelect div {
        background: #111827 !important;
        border: 1px solid #1e293b !important;
        border-radius: 8px !important;
        color: #f8fafc !important;
    }
    .stTextInput label, .stTextArea label, .stSelectbox label, .stMultiSelect label {
        font-family: 'Rajdhani', sans-serif;
        color: #94a3b8 !important;
    }

    .stSlider div[role="slider"] { background: linear-gradient(90deg, #00f5ff, #b24bf3); }
    .stCheckbox label { font-family: 'Rajdhani', sans-serif; color: #94a3b8; }
    .stDateInput input { background: #111827 !important; border: 1px solid #1e293b !important; }

    .stDataFrame { border: 1px solid #1e293b; border-radius: 12px; overflow: hidden; }
    .stDataFrame table { background: #111827 !important; }
    .stDataFrame thead th { background: #0f172a !important; color: #00f5ff !important; font-family: 'Orbitron', monospace; }
    .stDataFrame tbody td { color: #f8fafc !important; }

    .stProgress > div > div { background: #1e293b; }
    .stProgress > div > div > div { background: linear-gradient(90deg, #00f5ff, #b24bf3); }

    .stSuccess { background: rgba(16, 185, 129, 0.1) !important; border: 1px solid #10b981 !important; color: #10b981 !important; border-radius: 8px !important; }
    .stWarning { background: rgba(245, 158, 11, 0.1) !important; border: 1px solid #f59e0b !important; color: #f59e0b !important; border-radius: 8px !important; }
    .stError { background: rgba(239, 68, 68, 0.1) !important; border: 1px solid #ef4444 !important; color: #ef4444 !important; border-radius: 8px !important; }
    .stInfo { background: rgba(59, 130, 246, 0.1) !important; border: 1px solid #3b82f6 !important; color: #3b82f6 !important; border-radius: 8px !important; }

    section[data-testid="stSidebar"] h3 { font-family: 'Orbitron', monospace; color: #b24bf3; font-size: 0.85rem; margin-top: 1.2rem; }
    section[data-testid="stSidebar"] hr { border-color: #1e293b; }

    .analysis-card { background: linear-gradient(145deg, #111827, #0f172a); border: 1px solid #1e293b; border-radius: 12px; padding: 1.2rem; margin: 0.8rem 0; }
    .analysis-card p { font-family: 'Rajdhani', sans-serif; color: #94a3b8; line-height: 1.6; }

    .grid-bg {
        position: fixed; top: 0; left: 0; right: 0; bottom: 0;
        background-image: linear-gradient(rgba(0, 245, 255, 0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(0, 245, 255, 0.03) 1px, transparent 1px);
        background-size: 50px 50px; pointer-events: none; z-index: -1;
    }

    .welcome-container { text-align: center; padding: 2rem; max-width: 900px; margin: 0 auto; }
    .welcome-container h3 { font-family: 'Orbitron', monospace; color: #64748b; margin-bottom: 1rem; }
    .feature-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-top: 1.5rem; }
    .feature-card { background: linear-gradient(145deg, #111827, #0f172a); border: 1px solid #1e293b; border-radius: 12px; padding: 1rem; text-align: left; }
    .feature-card h4 { font-family: 'Orbitron', monospace; color: #00f5ff; font-size: 0.9rem; margin-bottom: 0.5rem; }
    .feature-card p { font-family: 'Rajdhani', sans-serif; color: #94a3b8; font-size: 0.85rem; }

    .footer { text-align: center; padding: 1.5rem; margin-top: 1.5rem; border-top: 1px solid #1e293b; }
    .footer p { font-family: 'Rajdhani', sans-serif; color: #64748b; margin: 0.2rem 0; }
    .footer .version { font-family: 'Orbitron', monospace; color: #00f5ff; font-size: 0.8rem; }

    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: #0a0e17; }
    ::-webkit-scrollbar-thumb { background: linear-gradient(180deg, #00f5ff, #b24bf3); border-radius: 4px; }
</style>

<div class="grid-bg"></div>
""", unsafe_allow_html=True)

# 标题
st.markdown('<h1 class="main-title">QUANTSTOCK PRO</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">AI-Powered Quantitative Trading Platform</p>', unsafe_allow_html=True)

# 创建标签页 - sidebar 放在外面
tab1, tab2 = st.tabs(["⚡ STRATEGY BACKTEST", "🤖 AI ANALYSIS"])

# ========== 侧边栏配置 - 放在 tabs 外面 ==========
with st.sidebar:
    st.markdown("## ⚙️ CONTROL PANEL")
    st.markdown("---")

    # 使用查询参数判断当前 tab
    # 默认显示回测配置
    st.markdown("### 📊 DATA INPUT")
    st.markdown("<small style='color:#64748b'>A股: 600519 | 港股: 00700 | 美股: AAPL</small>", unsafe_allow_html=True)

    symbols_input = st.text_area(
        "Stock Codes",
        value="600519\n000858\n000333\n601318\n600036",
        height=80,
        key="backtest_symbols",
        label_visibility="collapsed"
    )

    symbols = []
    for line in symbols_input.split('\n'):
        line = line.strip()
        if ',' in line:
            symbols.extend([s.strip() for s in line.split(',') if s.strip()])
        elif line:
            symbols.append(line)

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", value=datetime.now() - timedelta(days=365), key="backtest_start")
    with col2:
        end_date = st.date_input("End", value=datetime.now(), key="backtest_end")

    interval = st.selectbox("Frequency", options=["1d", "1wk", "1mo"], index=0, key="backtest_interval")

    st.markdown("### 🎯 STRATEGY")
    strategy_type = st.selectbox(
        "Strategy",
        options=["动量策略", "双动量策略", "均值回归策略", "多因子策略"],
        index=0, key="backtest_strategy", label_visibility="collapsed"
    )

    if strategy_type == "动量策略":
        lookback = st.slider("Lookback", 5, 250, 20, key="bt_lookback")
        holding = st.slider("Holding", 1, 60, 10, key="bt_holding")
        top_n = st.slider("Top N", 1, 20, 5, key="bt_top_n")
    elif strategy_type == "双动量策略":
        abs_lookback = st.slider("Abs Momentum", 5, 100, 20, key="bt_abs_lookback")
        rel_lookback = st.slider("Rel Momentum", 20, 250, 60, key="bt_rel_lookback")
        top_n = st.slider("Top N", 1, 20, 5, key="bt_top_n2")
    elif strategy_type == "均值回归策略":
        lookback = st.slider("Lookback", 5, 250, 20, key="bt_lookback2")
        zscore_threshold = st.slider("Z-Score", 1.0, 3.0, 2.0, 0.1, key="bt_zscore")
    else:
        top_n = st.slider("Top N", 1, 20, 5, key="bt_top_n3")
        rebalance_freq = st.selectbox("Rebalance", ["每日", "每周", "每月"], index=2, key="bt_rebalance")

    st.markdown("### 💰 CAPITAL")
    initial_capital = st.number_input("Capital", min_value=10000, max_value=100000000, value=1000000, step=10000, key="bt_capital")

    col1, col2 = st.columns(2)
    with col1:
        commission_rate = st.number_input("Commission%", min_value=0.0, max_value=1.0, value=0.03, step=0.01, key="bt_commission") / 100
    with col2:
        slippage_rate = st.number_input("Slippage%", min_value=0.0, max_value=1.0, value=0.01, step=0.01, key="bt_slippage") / 100

    st.markdown("### 🔢 FACTORS")
    factor_calculator = FactorCalculator()
    technical_factors = factor_calculator.list_factors(FactorType.TECHNICAL)
    default_technical = ["ma_20", "macd"]
    rsi_factors = [f for f in technical_factors if f.startswith("rsi_")]
    if rsi_factors:
        default_technical.append(rsi_factors[0])
    selected_technical = st.multiselect("Indicators", options=technical_factors, default=default_technical, key="bt_technical", label_visibility="collapsed")

    st.markdown("---")
    run_backtest = st.button("🚀 EXECUTE BACKTEST", type="primary", use_container_width=True, key="btn_backtest")

    # AI 分析配置
    st.markdown("---")
    st.markdown("## 🤖 AI ANALYSIS")
    st.markdown("---")

    analyze_symbols = st.text_input("Codes (comma)", value="600519,000858,000333", key="analyze_symbols", label_visibility="collapsed")
    analyze_stocks = [s.strip() for s in analyze_symbols.split(',') if s.strip()]

    col1, col2 = st.columns(2)
    with col1:
        analyze_start = st.date_input("Start Date", value=datetime.now() - timedelta(days=180), key="analyze_start")
    with col2:
        analyze_end = st.date_input("End Date", value=datetime.now(), key="analyze_end")

    enable_search = st.checkbox("Web Search", value=True, key="analyze_search")
    enable_chip = st.checkbox("Chip Data", value=True, key="analyze_chip")
    enable_realtime = st.checkbox("Real-time", value=True, key="analyze_realtime")

    enable_notify = st.checkbox("Notify", value=False, key="analyze_notify")
    if enable_notify:
        notify_channels = st.multiselect("Channels", options=["email", "wechat"], default=["email"], key="analyze_channels", label_visibility="collapsed")
    else:
        notify_channels = ["email"]

    st.markdown("---")
    run_analyze = st.button("🤖 START AI ANALYSIS", type="primary", use_container_width=True, key="btn_analyze")


# ========== 标签页1: 策略回测 ==========
with tab1:
    if run_backtest and symbols:
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.markdown("<p style='color:#00f5ff'>📥 Fetching data...</p>", unsafe_allow_html=True)
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
            else:
                status_text.markdown("<p style='color:#00f5ff'>🔢 Computing factors...</p>", unsafe_allow_html=True)
                calculator = FactorCalculator()
                selected_factors = selected_technical

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

                if strategy_type in ["动量策略", "双动量策略"]:
                    backtest_data = pd.DataFrame({s: d['Close'] for s, d in stock_data.items()})
                    mode_bt = BacktestMode.MULTI_STOCK
                else:
                    first_symbol = list(stock_data.keys())[0]
                    backtest_data = stock_data[first_symbol]
                    mode_bt = BacktestMode.SINGLE_STOCK

                status_text.markdown("<p style='color:#00f5ff'>⚡ Running backtest...</p>", unsafe_allow_html=True)
                backtester = Backtester(strategy=strategy, initial_capital=initial_capital, commission_rate=commission_rate, slippage_rate=slippage_rate)
                result = backtester.run(data=backtest_data, start_date=start_date.strftime("%Y-%m-%d"), end_date=end_date.strftime("%Y-%m-%d"), mode=mode_bt)

                progress_bar.empty()
                status_text.empty()

                st.markdown('<h2 class="section-header">📊 PERFORMANCE</h2>', unsafe_allow_html=True)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    total_return = result.performance_metrics.get('总收益率%', 0)
                    return_class = "positive" if total_return > 0 else "negative"
                    st.markdown(f'<div class="metric-card"><div class="metric-label">TOTAL RETURN</div><div class="metric-value {return_class}">{total_return:.2f}%</div></div>', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<div class="metric-card"><div class="metric-label">ANNUAL</div><div class="metric-value">{result.performance_metrics.get("年化收益率%", 0):.2f}%</div></div>', unsafe_allow_html=True)
                with col3:
                    sharpe = result.performance_metrics.get('夏普比率', 0)
                    st.markdown(f'<div class="metric-card"><div class="metric-label">SHARPE</div><div class="metric-value">{sharpe:.3f}</div></div>', unsafe_allow_html=True)
                with col4:
                    max_dd = result.performance_metrics.get('最大回撤%', 0)
                    st.markdown(f'<div class="metric-card"><div class="metric-label">MAX DD</div><div class="metric-value">{max_dd:.2f}%</div></div>', unsafe_allow_html=True)

                st.markdown('<h3 class="section-header">📈 METRICS</h3>', unsafe_allow_html=True)
                metrics_data = [
                    ["INITIAL", f"¥{result.performance_metrics.get('初始资金', 0):,.0f}"],
                    ["FINAL", f"¥{result.performance_metrics.get('最终权益', 0):,.0f}"],
                    ["WIN RATE", f"{result.performance_metrics.get('胜率%', 0):.2f}%"],
                    ["TRADES", f"{result.performance_metrics.get('总交易次数', 0)}"],
                ]
                st.dataframe(pd.DataFrame(metrics_data, columns=["METRIC", "VALUE"]), use_container_width=True, hide_index=True)

                st.markdown('<h3 class="section-header">📊 CHARTS</h3>', unsafe_allow_html=True)
                plotter = Plotter()
                fig1 = plotter.plot_equity_curve(result.equity_curve, result.benchmark_returns, title="EQUITY CURVE", show=False)
                fig1.update_layout(paper_bgcolor='#111827', plot_bgcolor='#111827', font_color='#94a3b8')
                st.plotly_chart(fig1, use_container_width=True)

                st.success("✓ BACKTEST COMPLETE")

        except Exception as e:
            st.error(f"✗ Error: {str(e)}")
            import traceback
            with st.expander("Details"):
                st.code(traceback.format_exc())
    else:
        st.markdown("""
        <div class="welcome-container">
            <h3>⚡ STRATEGY BACKTEST</h3>
            <p>Configure parameters in sidebar and click "EXECUTE BACKTEST"</p>
            <div class="feature-grid">
                <div class="feature-card"><h4>📊 Multi-Source Data</h4><p>A-share, HK, US stocks</p></div>
                <div class="feature-card"><h4>🎯 Strategies</h4><p>Momentum, Mean Reversion, Multi-Factor</p></div>
                <div class="feature-card"><h4>📈 Indicators</h4><p>MA, MACD, RSI, Bollinger</p></div>
                <div class="feature-card"><h4>💰 Risk Metrics</h4><p>Sharpe, Max Drawdown, Win Rate</p></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ========== 标签页2: AI智能分析 ==========
with tab2:
    if run_analyze and analyze_stocks:
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.markdown("<p style='color:#b24bf3'>🔧 Initializing...</p>", unsafe_allow_html=True)
            data_fetcher = EnhancedDataFetcher()
            search_manager = SearchAPIManager() if enable_search else None
            llm_analyzer = LLMAnalyzer()
            report_generator = ReportGenerator()
            notifier = Notifier() if enable_notify else None

            analyses_results = []

            for i, symbol in enumerate(analyze_stocks):
                st.markdown(f'<h2 class="section-header purple">📊 {symbol}</h2>', unsafe_allow_html=True)

                with st.spinner(f"Fetching {symbol}..."):
                    stock_data = data_fetcher.get_full_stock_data(
                        symbol=symbol,
                        start_date=analyze_start.strftime("%Y-%m-%d"),
                        end_date=analyze_end.strftime("%Y-%m-%d"),
                        include_realtime=enable_realtime,
                        include_chip=enable_chip,
                        include_institutional=False
                    )

                if stock_data.realtime:
                    rt = stock_data.realtime
                    change_class = "positive" if rt.change_pct > 0 else "negative"
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f'<div class="metric-card"><div class="metric-label">STOCK</div><div class="metric-value">{rt.name}</div></div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown(f'<div class="metric-card"><div class="metric-label">PRICE</div><div class="metric-value {change_class}">¥{rt.price:.2f}</div></div>', unsafe_allow_html=True)
                    with col3:
                        st.markdown(f'<div class="metric-card"><div class="metric-label">CHANGE</div><div class="metric-value {change_class}">{rt.change_pct:+.2f}%</div></div>', unsafe_allow_html=True)
                    stock_name = rt.name
                else:
                    stock_name = symbol

                with st.spinner("AI analyzing..."):
                    search_info = None
                    if search_manager:
                        search_info = asyncio.run(search_manager.search_stock_info(symbol=symbol, stock_name=stock_name))
                    analysis = asyncio.run(llm_analyzer.analyze_stock(stock_data=stock_data, search_info=search_info, stock_name=stock_name))

                rating_class = {"强烈推荐": "rating-strong-buy", "推荐": "rating-buy", "中性": "rating-neutral", "不推荐": "rating-sell"}.get(analysis.rating, "rating-neutral")

                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.markdown(f"**RATING**: <span class='rating-badge {rating_class}'>{analysis.rating}</span>", unsafe_allow_html=True)
                with col2:
                    if analysis.target_price:
                        st.metric("TARGET", f"¥{analysis.target_price:.2f}")
                with col3:
                    if analysis.stop_loss:
                        st.metric("STOP", f"¥{analysis.stop_loss:.2f}")

                tab_a, tab_b, tab_c, tab_d = st.tabs(["📊 Fundamental", "📈 Technical", "🎯 Chip", "💡 Advice"])
                with tab_a:
                    st.markdown(f"<div class='analysis-card'>{analysis.fundamental_analysis or 'N/A'}</div>", unsafe_allow_html=True)
                with tab_b:
                    st.markdown(f"<div class='analysis-card'>{analysis.technical_analysis or 'N/A'}</div>", unsafe_allow_html=True)
                with tab_c:
                    st.markdown(f"<div class='analysis-card'>{analysis.chip_analysis or 'N/A'}</div>", unsafe_allow_html=True)
                with tab_d:
                    st.markdown(f"<div class='analysis-card'>{analysis.operation_advice or 'N/A'}</div>", unsafe_allow_html=True)

                if analysis.risks:
                    st.warning("⚠️ " + ", ".join(analysis.risks))

                analyses_results.append(analysis)
                progress_bar.progress((i + 1) / len(analyze_stocks))

            progress_bar.empty()
            status_text.empty()

            if len(analyses_results) > 1:
                st.markdown('<h2 class="section-header purple">📊 SUMMARY</h2>', unsafe_allow_html=True)
                summary_data = [[f"{a.stock_name} ({a.symbol})", a.rating, f"¥{a.target_price:.2f}" if a.target_price else "-", a.model_used] for a in analyses_results]
                st.dataframe(pd.DataFrame(summary_data, columns=["STOCK", "RATING", "TARGET", "MODEL"]), use_container_width=True, hide_index=True)

            st.success("✓ AI ANALYSIS COMPLETE")

        except Exception as e:
            st.error(f"✗ Error: {str(e)}")
            import traceback
            with st.expander("Details"):
                st.code(traceback.format_exc())
    else:
        st.markdown("""
        <div class="welcome-container">
            <h3>🤖 AI ANALYSIS</h3>
            <p>Configure in sidebar and click "START AI ANALYSIS"</p>
            <div class="feature-grid">
                <div class="feature-card"><h4>📊 Fundamental</h4><p>Business, industry, financials</p></div>
                <div class="feature-card"><h4>📈 Technical</h4><p>Trends, support/resistance</p></div>
                <div class="feature-card"><h4>🎯 Chip Analysis</h4><p>Distribution, cost basis</p></div>
                <div class="feature-card"><h4>📰 News</h4><p>Sentiment, announcements</p></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# 页脚
st.markdown("""
<div class="footer">
    <p class="version">QUANTSTOCK PRO v2.4</p>
    <p>Strategy Backtest • AI Analysis</p>
</div>
""", unsafe_allow_html=True)
