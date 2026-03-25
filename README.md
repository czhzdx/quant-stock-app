# 量化选股软件 (Quantitative Stock Selection System)

一个基于Python的量化选股软件，支持多因子模型、技术指标分析和策略回测。

## 功能特性

### 核心功能
- 📊 **数据获取**: 支持从多个数据源获取股票数据
- 🔢 **因子计算**: 内置100+技术指标和基本面因子
- 📈 **策略开发**: 支持自定义选股策略
- 🔄 **回测引擎**: 完整的策略回测系统
- 📊 **可视化**: 交互式图表和报告
- 🤖 **机器学习**: 集成机器学习模型进行预测

### 支持的策略类型
1. **多因子模型** - 基于多个因子的综合评分
2. **技术指标策略** - 基于技术分析指标
3. **动量策略** - 追涨杀跌
4. **均值回归策略** - 价格回归策略
5. **机器学习策略** - 基于AI的预测模型

## 项目结构

```
quant_stock_selection/
├── data/                    # 数据存储
│   ├── raw/                # 原始数据
│   ├── processed/          # 处理后的数据
│   └── cache/              # 缓存数据
├── src/                    # 源代码
│   ├── data/               # 数据模块
│   ├── factors/            # 因子计算
│   ├── strategies/         # 策略模块
│   ├── backtest/           # 回测引擎
│   ├── visualization/      # 可视化
│   └── utils/              # 工具函数
├── config/                 # 配置文件
├── notebooks/              # Jupyter notebooks
├── tests/                  # 测试文件
├── requirements.txt        # 依赖包
├── main.py                 # 主程序
└── web_app.py              # Web界面
```

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行示例策略
```bash
python main.py --strategy momentum --start 2023-01-01 --end 2024-01-01
```

### 3. 启动Web界面
```bash
streamlit run web_app.py
```

## 配置说明

### 数据源配置
- **默认数据源**: Yahoo Finance (yfinance)
- **备用数据源**: 东方财富、新浪财经
- **数据频率**: 日线、周线、月线

### 因子库
- 技术指标: MA, MACD, RSI, Bollinger Bands等
- 基本面因子: PE, PB, ROE, 营收增长率等
- 量价因子: 成交量、换手率、资金流等
- 市场因子: Beta、波动率、相关性等

## 使用示例

### 基础使用
```python
from src.data.data_fetcher import DataFetcher
from src.strategies.momentum import MomentumStrategy

# 获取数据
fetcher = DataFetcher()
data = fetcher.get_stock_data('AAPL', '2023-01-01', '2024-01-01')

# 运行策略
strategy = MomentumStrategy()
signals = strategy.generate_signals(data)

# 回测
from src.backtest.backtester import Backtester
backtester = Backtester(strategy)
results = backtester.run(data)
```

### 自定义策略
```python
from src.strategies.base import BaseStrategy

class MyStrategy(BaseStrategy):
    def generate_signals(self, data):
        # 自定义选股逻辑
        signals = pd.Series(index=data.index)
        # ... 你的策略逻辑
        return signals
```

## 性能指标

回测系统提供以下性能指标：
- 年化收益率
- 夏普比率
- 最大回撤
- 胜率
- 盈亏比
- Alpha/Beta系数

## 数据更新

### 自动更新
```bash
python update_data.py --mode daily
```

### 手动更新
```bash
python update_data.py --symbols AAPL,MSFT,GOOGL --start 2024-01-01
```

## 注意事项

1. **数据延迟**: 免费数据源通常有15分钟延迟
2. **回测过拟合**: 注意避免在历史数据上过度优化
3. **交易成本**: 回测时考虑手续费和滑点
4. **风险控制**: 设置止损止盈和仓位管理

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！

## 联系方式

如有问题，请提交Issue或联系项目维护者。