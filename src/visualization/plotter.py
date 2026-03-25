"""可视化模块"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from ..backtest.backtester import BacktestResult
from ..utils.config_loader import get_config

logger = logging.getLogger(__name__)


class Plotter:
    """可视化绘制器"""

    def __init__(self, theme: str = "plotly_white"):
        """
        初始化可视化绘制器

        Args:
            theme: 主题风格
        """
        self.theme = theme
        self.config = get_config()  # 获取整个配置字典
        self.color_scheme = self.config.get("visualization", {}).get("color_scheme", "viridis")

        # 设置matplotlib风格
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette(self.color_scheme)

    def plot_equity_curve(
        self,
        equity_curve: pd.Series,
        benchmark_curve: Optional[pd.Series] = None,
        title: str = "权益曲线",
        show: bool = True,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        绘制权益曲线

        Args:
            equity_curve: 策略权益曲线
            benchmark_curve: 基准权益曲线
            title: 图表标题
            show: 是否显示图表
            save_path: 保存路径

        Returns:
            Plotly图表对象
        """
        fig = go.Figure()

        # 绘制策略权益曲线
        fig.add_trace(go.Scatter(
            x=equity_curve.index,
            y=equity_curve.values,
            mode='lines',
            name='策略净值',
            line=dict(color='blue', width=2),
            hovertemplate='日期: %{x}<br>净值: %{y:.2f}<extra></extra>'
        ))

        # 绘制基准权益曲线（如果有）
        if benchmark_curve is not None:
            # 对齐起始点
            benchmark_aligned = benchmark_curve / benchmark_curve.iloc[0] * equity_curve.iloc[0]
            fig.add_trace(go.Scatter(
                x=benchmark_aligned.index,
                y=benchmark_aligned.values,
                mode='lines',
                name='基准净值',
                line=dict(color='red', width=2, dash='dash'),
                hovertemplate='日期: %{x}<br>净值: %{y:.2f}<extra></extra>'
            ))

        # 计算最大回撤
        drawdown = self._calculate_drawdown(equity_curve)
        if not drawdown.empty:
            fig.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                mode='lines',
                name='回撤',
                line=dict(color='orange', width=1),
                yaxis='y2',
                hovertemplate='日期: %{x}<br>回撤: %{y:.2%}<extra></extra>'
            ))

        # 更新布局
        fig.update_layout(
            title=title,
            xaxis_title="日期",
            yaxis_title="净值",
            yaxis2=dict(
                title="回撤",
                overlaying='y',
                side='right',
                tickformat='.1%'
            ),
            hovermode='x unified',
            template=self.theme,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            height=600
        )

        if show:
            fig.show()

        if save_path:
            fig.write_html(save_path)
            logger.info(f"图表已保存到: {save_path}")

        return fig

    def plot_returns_distribution(
        self,
        returns: pd.Series,
        title: str = "收益率分布",
        show: bool = True,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        绘制收益率分布

        Args:
            returns: 收益率序列
            title: 图表标题
            show: 是否显示图表
            save_path: 保存路径

        Returns:
            Plotly图表对象
        """
        fig = go.Figure()

        # 直方图
        fig.add_trace(go.Histogram(
            x=returns.values,
            nbinsx=50,
            name='收益率分布',
            marker_color='skyblue',
            opacity=0.7,
            hovertemplate='收益率: %{x:.2%}<br>频数: %{y}<extra></extra>'
        ))

        # 添加正态分布曲线
        if len(returns) > 1:
            x_norm = np.linspace(returns.min(), returns.max(), 100)
            mean = returns.mean()
            std = returns.std()
            y_norm = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_norm - mean)/std)**2)
            y_norm = y_norm * len(returns) * (returns.max() - returns.min()) / 50

            fig.add_trace(go.Scatter(
                x=x_norm,
                y=y_norm,
                mode='lines',
                name='正态分布',
                line=dict(color='red', width=2),
                hovertemplate='收益率: %{x:.2%}<br>密度: %{y:.4f}<extra></extra>'
            ))

        # 更新布局
        fig.update_layout(
            title=title,
            xaxis_title="收益率",
            yaxis_title="频数",
            hovermode='x unified',
            template=self.theme,
            bargap=0.1,
            height=500
        )

        # 添加统计信息注释
        stats_text = (
            f"均值: {returns.mean():.4%}<br>"
            f"标准差: {returns.std():.4%}<br>"
            f"偏度: {returns.skew():.4f}<br>"
            f"峰度: {returns.kurtosis():.4f}"
        )

        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text=stats_text,
            showarrow=False,
            font=dict(size=12),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            borderpad=4
        )

        if show:
            fig.show()

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_monthly_returns(
        self,
        returns: pd.Series,
        title: str = "月度收益率热力图",
        show: bool = True,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        绘制月度收益率热力图

        Args:
            returns: 收益率序列
            title: 图表标题
            show: 是否显示图表
            save_path: 保存路径

        Returns:
            Plotly图表对象
        """
        # 转换为月度收益率
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)

        # 创建月度数据透视表
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        monthly_df = pd.DataFrame({
            'returns': monthly_returns.values,
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month
        }, index=monthly_returns.index)

        pivot_table = monthly_df.pivot_table(
            values='returns',
            index='year',
            columns='month',
            aggfunc='last'
        )

        # 创建热力图
        fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values * 100,  # 转换为百分比
            x=[f'{m}月' for m in pivot_table.columns],
            y=pivot_table.index.astype(str),
            colorscale='RdYlGn',
            zmid=0,
            text=pivot_table.values * 100,
            texttemplate='%{text:.1f}%',
            textfont={"size": 10},
            hovertemplate='年份: %{y}<br>月份: %{x}<br>收益率: %{z:.2f}%<extra></extra>'
        ))

        # 更新布局
        fig.update_layout(
            title=title,
            xaxis_title="月份",
            yaxis_title="年份",
            template=self.theme,
            height=500
        )

        if show:
            fig.show()

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_rolling_metrics(
        self,
        equity_curve: pd.Series,
        window: int = 60,
        title: str = "滚动指标",
        show: bool = True,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        绘制滚动指标

        Args:
            equity_curve: 权益曲线
            window: 滚动窗口大小
            title: 图表标题
            show: 是否显示图表
            save_path: 保存路径

        Returns:
            Plotly图表对象
        """
        # 计算滚动指标
        returns = equity_curve.pct_change().fillna(0)

        # 滚动夏普比率
        rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
        rolling_sharpe = rolling_sharpe.replace([np.inf, -np.inf], np.nan).fillna(0)

        # 滚动波动率
        rolling_vol = returns.rolling(window).std() * np.sqrt(252) * 100

        # 滚动最大回撤
        rolling_drawdown = self._calculate_rolling_drawdown(equity_curve, window)

        # 创建子图
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('滚动夏普比率', '滚动波动率 (%)', '滚动最大回撤 (%)'),
            vertical_spacing=0.1
        )

        # 滚动夏普比率
        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe.values,
                mode='lines',
                name='夏普比率',
                line=dict(color='blue', width=2),
                hovertemplate='日期: %{x}<br>夏普比率: %{y:.3f}<extra></extra>'
            ),
            row=1, col=1
        )

        # 滚动波动率
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol.values,
                mode='lines',
                name='波动率',
                line=dict(color='green', width=2),
                hovertemplate='日期: %{x}<br>波动率: %{y:.2f}%<extra></extra>'
            ),
            row=2, col=1
        )

        # 滚动最大回撤
        fig.add_trace(
            go.Scatter(
                x=rolling_drawdown.index,
                y=rolling_drawdown.values * 100,
                mode='lines',
                name='最大回撤',
                line=dict(color='red', width=2),
                hovertemplate='日期: %{x}<br>最大回撤: %{y:.2f}%<extra></extra>'
            ),
            row=3, col=1
        )

        # 更新布局
        fig.update_layout(
            title=title,
            height=800,
            showlegend=False,
            template=self.theme,
            hovermode='x unified'
        )

        fig.update_xaxes(title_text="日期", row=3, col=1)
        fig.update_yaxes(title_text="夏普比率", row=1, col=1)
        fig.update_yaxes(title_text="波动率 (%)", row=2, col=1)
        fig.update_yaxes(title_text="最大回撤 (%)", row=3, col=1)

        if show:
            fig.show()

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_trades(
        self,
        price_data: pd.Series,
        trades: pd.DataFrame,
        title: str = "交易信号",
        show: bool = True,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        绘制交易信号图

        Args:
            price_data: 价格序列
            trades: 交易记录DataFrame
            title: 图表标题
            show: 是否显示图表
            save_path: 保存路径

        Returns:
            Plotly图表对象
        """
        fig = go.Figure()

        # 绘制价格曲线
        fig.add_trace(go.Scatter(
            x=price_data.index,
            y=price_data.values,
            mode='lines',
            name='价格',
            line=dict(color='black', width=1),
            hovertemplate='日期: %{x}<br>价格: %{y:.2f}<extra></extra>'
        ))

        if not trades.empty:
            # 买入信号
            buy_trades = trades[trades['type'] == 'BUY']
            if not buy_trades.empty:
                fig.add_trace(go.Scatter(
                    x=buy_trades.index,
                    y=price_data.reindex(buy_trades.index).values,
                    mode='markers',
                    name='买入',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color='green',
                        line=dict(width=2, color='darkgreen')
                    ),
                    hovertemplate='日期: %{x}<br>价格: %{y:.2f}<br>类型: 买入<extra></extra>'
                ))

            # 卖出信号
            sell_trades = trades[trades['type'] == 'SELL']
            if not sell_trades.empty:
                fig.add_trace(go.Scatter(
                    x=sell_trades.index,
                    y=price_data.reindex(sell_trades.index).values,
                    mode='markers',
                    name='卖出',
                    marker=dict(
                        symbol='triangle-down',
                        size=12,
                        color='red',
                        line=dict(width=2, color='darkred')
                    ),
                    hovertemplate='日期: %{x}<br>价格: %{y:.2f}<br>类型: 卖出<extra></extra>'
                ))

        # 更新布局
        fig.update_layout(
            title=title,
            xaxis_title="日期",
            yaxis_title="价格",
            hovermode='x unified',
            template=self.theme,
            height=600
        )

        if show:
            fig.show()

        if save_path:
            fig.write_html(save_path)

        return fig

    def plot_performance_summary(
        self,
        result: BacktestResult,
        title: str = "策略性能摘要",
        show: bool = True,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        绘制策略性能摘要图

        Args:
            result: 回测结果
            title: 图表标题
            show: 是否显示图表
            save_path: 保存路径

        Returns:
            Plotly图表对象
        """
        metrics = result.performance_metrics

        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('权益曲线', '收益率分布', '月度收益率', '滚动指标'),
            specs=[
                [{"type": "scatter"}, {"type": "histogram"}],
                [{"type": "heatmap"}, {"type": "scatter"}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )

        # 1. 权益曲线
        fig.add_trace(
            go.Scatter(
                x=result.equity_curve.index,
                y=result.equity_curve.values,
                mode='lines',
                name='净值',
                line=dict(color='blue', width=2),
                hovertemplate='日期: %{x}<br>净值: %{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )

        # 2. 收益率分布
        fig.add_trace(
            go.Histogram(
                x=result.returns.values,
                nbinsx=30,
                name='收益率分布',
                marker_color='skyblue',
                opacity=0.7,
                hovertemplate='收益率: %{x:.2%}<br>频数: %{y}<extra></extra>'
            ),
            row=1, col=2
        )

        # 3. 月度收益率热力图（简化版）
        monthly_returns = result.returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        monthly_df = pd.DataFrame({
            'returns': monthly_returns.values,
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month
        }, index=monthly_returns.index)

        pivot_table = monthly_df.pivot_table(
            values='returns',
            index='year',
            columns='month',
            aggfunc='last'
        )

        fig.add_trace(
            go.Heatmap(
                z=pivot_table.values * 100,
                x=[f'{m}月' for m in pivot_table.columns],
                y=pivot_table.index.astype(str),
                colorscale='RdYlGn',
                zmid=0,
                hovertemplate='年份: %{y}<br>月份: %{x}<br>收益率: %{z:.2f}%<extra></extra>'
            ),
            row=2, col=1
        )

        # 4. 滚动夏普比率
        rolling_sharpe = result.returns.rolling(60).mean() / result.returns.rolling(60).std() * np.sqrt(252)
        rolling_sharpe = rolling_sharpe.replace([np.inf, -np.inf], np.nan).fillna(0)

        fig.add_trace(
            go.Scatter(
                x=rolling_sharpe.index,
                y=rolling_sharpe.values,
                mode='lines',
                name='滚动夏普',
                line=dict(color='purple', width=2),
                hovertemplate='日期: %{x}<br>夏普比率: %{y:.3f}<extra></extra>'
            ),
            row=2, col=2
        )

        # 更新布局
        fig.update_layout(
            title=title,
            height=800,
            showlegend=False,
            template=self.theme,
            hovermode='x unified'
        )

        # 添加性能指标表格
        key_metrics = [
            ('总收益率%', f"{metrics.get('总收益率%', 0):.2f}"),
            ('年化收益率%', f"{metrics.get('年化收益率%', 0):.2f}"),
            ('夏普比率', f"{metrics.get('夏普比率', 0):.3f}"),
            ('最大回撤%', f"{metrics.get('最大回撤%', 0):.2f}"),
            ('胜率%', f"{metrics.get('胜率%', 0):.2f}"),
            ('总交易次数', f"{metrics.get('总交易次数', 0)}")
        ]

        metrics_text = "<br>".join([f"{k}: {v}" for k, v in key_metrics])

        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text=metrics_text,
            showarrow=False,
            font=dict(size=11),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            align="left"
        )

        if show:
            fig.show()

        if save_path:
            fig.write_html(save_path)

        return fig

    def _calculate_drawdown(self, equity_curve: pd.Series) -> pd.Series:
        """计算回撤"""
        cumulative = (1 + equity_curve.pct_change().fillna(0)).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = cumulative / running_max - 1
        return drawdown

    def _calculate_rolling_drawdown(self, equity_curve: pd.Series, window: int) -> pd.Series:
        """计算滚动最大回撤"""
        returns = equity_curve.pct_change().fillna(0)
        rolling_drawdown = pd.Series(index=returns.index, dtype=float)

        for i in range(window, len(returns)):
            window_returns = returns.iloc[i-window:i]
            cumulative = (1 + window_returns).cumprod()
            max_dd = (cumulative / cumulative.expanding().max() - 1).min()
            rolling_drawdown.iloc[i] = max_dd

        return rolling_drawdown.fillna(0)

    def save_all_plots(
        self,
        result: BacktestResult,
        price_data: Optional[pd.Series] = None,
        output_dir: str = "./plots"
    ):
        """
        保存所有图表

        Args:
            result: 回测结果
            price_data: 价格数据（用于交易信号图）
            output_dir: 输出目录
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 1. 权益曲线图
        self.plot_equity_curve(
            result.equity_curve,
            result.benchmark_returns,
            title="策略权益曲线",
            show=False,
            save_path=os.path.join(output_dir, f"equity_curve_{timestamp}.html")
        )

        # 2. 收益率分布图
        self.plot_returns_distribution(
            result.returns,
            title="收益率分布",
            show=False,
            save_path=os.path.join(output_dir, f"returns_distribution_{timestamp}.html")
        )

        # 3. 月度收益率热力图
        self.plot_monthly_returns(
            result.returns,
            title="月度收益率热力图",
            show=False,
            save_path=os.path.join(output_dir, f"monthly_returns_{timestamp}.html")
        )

        # 4. 滚动指标图
        self.plot_rolling_metrics(
            result.equity_curve,
            title="滚动指标",
            show=False,
            save_path=os.path.join(output_dir, f"rolling_metrics_{timestamp}.html")
        )

        # 5. 交易信号图（如果有价格数据和交易记录）
        if price_data is not None and not result.trades.empty:
            self.plot_trades(
                price_data,
                result.trades,
                title="交易信号",
                show=False,
                save_path=os.path.join(output_dir, f"trades_{timestamp}.html")
            )

        # 6. 性能摘要图
        self.plot_performance_summary(
            result,
            title="策略性能摘要",
            show=False,
            save_path=os.path.join(output_dir, f"performance_summary_{timestamp}.html")
        )

        logger.info(f"所有图表已保存到: {output_dir}")


# 测试函数
def test_plotter():
    """测试可视化模块"""
    import logging
    logging.basicConfig(level=logging.INFO)

    # 创建示例数据
    dates = pd.date_range('2023-01-01', periods=252, freq='B')  # 交易日
    np.random.seed(42)

    # 创建权益曲线
    returns = np.random.randn(len(dates)) * 0.01
    equity_curve = pd.Series(1000000 * (1 + returns).cumprod(), index=dates)

    # 创建基准曲线
    benchmark_returns = np.random.randn(len(dates)) * 0.008 + 0.0002
    benchmark_curve = pd.Series(1000000 * (1 + benchmark_returns).cumprod(), index=dates)

    # 创建交易记录
    trades = pd.DataFrame({
        'date': dates[[50, 100, 150, 200]],
        'type': ['BUY', 'SELL', 'BUY', 'SELL'],
        'price': [equity_curve.iloc[50], equity_curve.iloc[100],
                  equity_curve.iloc[150], equity_curve.iloc[200]] / 10000,
        'quantity': [100, 100, 150, 150],
        'value': [10000, 10000, 15000, 15000]
    }).set_index('date')

    # 创建回测结果
    result = BacktestResult(
        equity_curve=equity_curve,
        returns=pd.Series(returns, index=dates),
        positions=pd.DataFrame(),
        trades=trades,
        performance_metrics={
            '总收益率%': 15.5,
            '年化收益率%': 12.3,
            '夏普比率': 1.2,
            '最大回撤%': -8.7,
            '胜率%': 55.6,
            '盈亏比': 1.8,
            '总交易次数': 45,
            'Alpha': 3.2,
            'Beta': 0.95
        },
        benchmark_returns=pd.Series(benchmark_returns, index=dates)
    )

    # 测试可视化
    plotter = Plotter()

    print("测试权益曲线图...")
    fig1 = plotter.plot_equity_curve(
        equity_curve,
        benchmark_curve,
        title="测试权益曲线",
        show=False
    )
    print(f"权益曲线图创建成功: {type(fig1)}")

    print("\n测试收益率分布图...")
    fig2 = plotter.plot_returns_distribution(
        result.returns,
        title="测试收益率分布",
        show=False
    )
    print(f"收益率分布图创建成功: {type(fig2)}")

    print("\n测试月度收益率热力图...")
    fig3 = plotter.plot_monthly_returns(
        result.returns,
        title="测试月度收益率",
        show=False
    )
    print(f"月度收益率热力图创建成功: {type(fig3)}")

    print("\n测试滚动指标图...")
    fig4 = plotter.plot_rolling_metrics(
        equity_curve,
        title="测试滚动指标",
        show=False
    )
    print(f"滚动指标图创建成功: {type(fig4)}")

    print("\n测试交易信号图...")
    price_data = pd.Series(np.random.randn(len(dates)).cumsum() + 100, index=dates)
    fig5 = plotter.plot_trades(
        price_data,
        trades,
        title="测试交易信号",
        show=False
    )
    print(f"交易信号图创建成功: {type(fig5)}")

    print("\n测试性能摘要图...")
    fig6 = plotter.plot_performance_summary(
        result,
        title="测试性能摘要",
        show=False
    )
    print(f"性能摘要图创建成功: {type(fig6)}")

    print("\n测试保存所有图表...")
    plotter.save_all_plots(result, price_data, output_dir="./test_plots")
    print("图表保存完成")


if __name__ == "__main__":
    test_plotter()