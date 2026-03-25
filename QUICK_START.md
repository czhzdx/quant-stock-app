# 量化选股软件 - 快速启动指南

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行示例
```bash
python examples/example_usage.py
```

### 3. 启动Web应用
```bash
streamlit run web_app.py
```

### 4. 使用命令行
```bash
# 查看帮助
python main.py --help

# 运行动量策略回测
python main.py --symbols AAPL MSFT GOOGL --start-date 2023-01-01 --end-date 2024-01-01 --strategy momentum --plot

# 运行均值回归策略
python main.py --symbols 000001.SZ --start-date 2023-01-01 --end-date 2024-01-01 --strategy mean_reversion

# 运行多因子策略
python main.py --symbols AAPL MSFT GOOGL AMZN --start-date 2023-01-01 --end-date 2024-01-01 --strategy multi_factor --top-n 3
```

## 📁 项目结构

```
quant_stock_selection/
├── src/                    # 源代码
│   ├── data/              # 数据模块
│   ├── factors/           # 因子计算
│   ├── strategies/        # 策略模块
│   ├── backtest/          # 回测引擎
│   ├── visualization/     # 可视化
│   └── utils/            # 工具函数
├── config/               # 配置文件
├── examples/             # 使用示例
├── requirements.txt      # 依赖包
├── main.py              # 命令行主程序
├── web_app.py           # Web界面
└── setup.py             # 安装脚本
```

## 🔧 配置说明

编辑 `config/config.yaml` 文件来配置系统：

```yaml
# 数据源配置
data_sources:
  yahoo_finance:
    enabled: true
    timeout: 30
    retry_count: 3
    cache_days: 7

# 股票池配置
stock_pool:
  csi300: true          # 沪深300成分股
  csi500: true          # 中证500成分股
  custom_stocks: []     # 自定义股票列表

# 策略配置
strategies:
  momentum:
    enabled: true
    lookback_period: 20
    holding_period: 10
    top_n: 5
```

## 📊 支持的策略

### 1. 动量策略 (MomentumStrategy)
- 基于过去N期收益率排序
- 选择前N只股票持有M期
- 定期调仓

### 2. 双动量策略 (DualMomentumStrategy)
- 绝对动量筛选 + 相对动量排名
- 双重筛选机制
- 风险控制更好

### 3. 均值回归策略 (MeanReversionStrategy)
- 基于Z分数判断超买超卖
- 布林带、RSI等指标
- 反转交易信号

### 4. 多因子策略 (MultiFactorStrategy)
- 多因子加权评分
- 支持自定义因子权重
- 定期再平衡

## 📈 性能指标

系统计算以下性能指标：
- ✅ 总收益率、年化收益率
- ✅ 夏普比率、索提诺比率、卡尔玛比率
- ✅ 最大回撤、波动率
- ✅ 胜率、盈亏比
- ✅ Alpha、Beta系数
- ✅ 总交易次数、盈利交易次数

## 🎨 可视化功能

Web应用提供6种专业图表：
1. **权益曲线图** - 策略净值 vs 基准
2. **收益率分布图** - 收益分布直方图
3. **月度收益率热力图** - 月度表现可视化
4. **滚动指标图** - 滚动夏普比率、波动率、最大回撤
5. **交易信号图** - 买卖点标记
6. **性能摘要图** - 综合性能指标

## 🔍 故障排除

### 常见问题

1. **Yahoo Finance API限制**
   - 问题：`Too Many Requests. Rate limited.`
   - 解决：启用缓存，减少请求频率

2. **配置加载错误**
   - 问题：`TypeError: get_config() missing 1 required positional argument`
   - 解决：已修复，确保使用最新代码

3. **依赖安装失败**
   - 问题：`ModuleNotFoundError`
   - 解决：`pip install -r requirements.txt`

4. **Web应用无法启动**
   - 问题：`streamlit: command not found`
   - 解决：`pip install streamlit`

### 调试模式

启用详细日志：
```bash
python main.py --verbose
```

## 📚 扩展开发

### 添加新因子
1. 在 `src/factors/factor_calculator.py` 中添加计算方法
2. 在 `_register_factors` 方法中注册因子
3. 在配置文件中启用因子

### 添加新策略
1. 继承 `BaseStrategy` 类
2. 实现 `generate_signals` 方法
3. 在 `strategies/` 目录下创建新文件

### 添加新数据源
1. 在 `src/data/data_fetcher.py` 中添加数据源类
2. 实现数据获取方法
3. 在配置文件中配置数据源

## 📞 支持

如果遇到问题：
1. 检查日志文件 `quant_stock.log`
2. 查看示例代码 `examples/example_usage.py`
3. 运行测试脚本验证功能

## 📄 许可证

MIT License - 详见 LICENSE 文件

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request