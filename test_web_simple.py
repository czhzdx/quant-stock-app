"""简单测试Web应用功能"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 模拟Streamlit的st对象
class MockStreamlit:
    def __init__(self):
        self.messages = []

    def set_page_config(self, **kwargs):
        self.messages.append(f"set_page_config: {kwargs}")

    def markdown(self, text, **kwargs):
        self.messages.append(f"markdown: {text[:50]}...")

    def sidebar(self):
        return self

    def title(self, text):
        self.messages.append(f"title: {text}")

    def text_area(self, label, value, **kwargs):
        self.messages.append(f"text_area: {label}")
        return value

    def date_input(self, label, value, **kwargs):
        self.messages.append(f"date_input: {label}")
        return value

    def selectbox(self, label, options, **kwargs):
        self.messages.append(f"selectbox: {label}")
        return options[kwargs.get('index', 0)]

    def slider(self, label, min_value, max_value, value, **kwargs):
        self.messages.append(f"slider: {label}")
        return value

    def number_input(self, label, **kwargs):
        self.messages.append(f"number_input: {label}")
        return kwargs.get('value', 0)

    def multiselect(self, label, options, default, **kwargs):
        self.messages.append(f"multiselect: {label}")
        return default

    def checkbox(self, label, value, **kwargs):
        self.messages.append(f"checkbox: {label}")
        return value

    def button(self, label, **kwargs):
        self.messages.append(f"button: {label}")
        return True

    def progress(self, value):
        self.messages.append(f"progress: {value}")
        return self

    def empty(self):
        return self

    def text(self, text):
        self.messages.append(f"text: {text}")

    def success(self, text):
        self.messages.append(f"success: {text}")

    def warning(self, text):
        self.messages.append(f"warning: {text}")

    def error(self, text):
        self.messages.append(f"error: {text}")

    def info(self, text):
        self.messages.append(f"info: {text}")

    def dataframe(self, data):
        self.messages.append(f"dataframe: shape={data.shape if hasattr(data, 'shape') else 'N/A'}")

    def plotly_chart(self, fig, **kwargs):
        self.messages.append(f"plotly_chart: created")

    def download_button(self, **kwargs):
        self.messages.append(f"download_button")

# 替换streamlit
sys.modules['streamlit'] = MockStreamlit()
st = MockStreamlit()

print("=" * 80)
print("测试Web应用逻辑（不启动服务器）")
print("=" * 80)

# 导入Web应用的主要函数进行测试
try:
    # 模拟用户输入
    symbols_input = "AAPL\nMSFT\nGOOGL\nAMZN\nTSLA"
    start_date = "2023-01-01"
    end_date = "2024-01-01"
    interval = "1d"
    use_mock_data = True
    strategy_type = "动量策略"
    lookback = 20
    holding = 10
    top_n = 5
    initial_capital = 1000000
    commission_rate = 0.0003
    slippage_rate = 0.0001

    print(f"模拟用户输入:")
    print(f"  股票代码: {symbols_input}")
    print(f"  开始日期: {start_date}")
    print(f"  结束日期: {end_date}")
    print(f"  数据频率: {interval}")
    print(f"  使用模拟数据: {use_mock_data}")
    print(f"  策略类型: {strategy_type}")
    print(f"  回顾期: {lookback}")
    print(f"  持有期: {holding}")
    print(f"  选择前N只: {top_n}")
    print(f"  初始资金: {initial_capital}")
    print(f"  手续费率: {commission_rate}")
    print(f"  滑点率: {slippage_rate}")

    # 测试数据获取
    print("\n测试数据获取...")
    try:
        from src.data.data_fetcher_fixed import DataFetcherFixed
        fetcher = DataFetcherFixed(use_mock_data=use_mock_data)

        # 解析股票代码
        symbols = []
        for line in symbols_input.split('\n'):
            line = line.strip()
            if ',' in line:
                symbols.extend([s.strip() for s in line.split(',') if s.strip()])
            elif line:
                symbols.append(line)

        print(f"  解析到股票代码: {symbols}")

        # 获取数据
        stock_data = {}
        for symbol in symbols[:2]:  # 只测试前2个
            data = fetcher.get_stock_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval=interval,
                adjust=True
            )
            if data is not None and not data.empty:
                stock_data[symbol] = data
                print(f"  [OK] 获取 {symbol} 数据成功: {data.shape}")
            else:
                print(f"  [ERROR] 获取 {symbol} 数据失败")

        if stock_data:
            print(f"  成功获取 {len(stock_data)} 只股票数据")
        else:
            print("  [ERROR] 没有获取到任何股票数据")

    except Exception as e:
        print(f"  [ERROR] 数据获取测试失败: {e}")
        import traceback
        traceback.print_exc()

    # 测试因子计算
    print("\n测试因子计算...")
    try:
        from src.factors.factor_calculator import FactorCalculator, FactorType

        if stock_data:
            calculator = FactorCalculator()
            symbol = list(stock_data.keys())[0]
            data = stock_data[symbol]

            # 获取技术指标因子
            technical_factors = calculator.list_factors(FactorType.TECHNICAL)
            selected_factors = technical_factors[:3]  # 只测试前3个

            factors = calculator.calculate_multiple_factors(data, selected_factors)
            print(f"  [OK] 因子计算成功")
            print(f"       计算的因子: {selected_factors}")
            print(f"       因子数据形状: {factors.shape}")
        else:
            print("  [SKIP] 没有数据可测试因子计算")

    except Exception as e:
        print(f"  [ERROR] 因子计算测试失败: {e}")

    # 测试策略
    print("\n测试策略...")
    try:
        from src.strategies.momentum import MomentumStrategy

        if strategy_type == "动量策略":
            strategy = MomentumStrategy(
                lookback_period=lookback,
                holding_period=holding,
                top_n=top_n
            )
            print(f"  [OK] 动量策略创建成功")
            print(f"       策略名称: {strategy.name}")
            print(f"       策略参数: {strategy.params}")
        else:
            print(f"  [SKIP] 策略类型 '{strategy_type}' 未测试")

    except Exception as e:
        print(f"  [ERROR] 策略测试失败: {e}")

    # 测试回测
    print("\n测试回测...")
    try:
        from src.backtest.backtester import Backtester, BacktestMode

        if stock_data and 'strategy' in locals():
            # 准备回测数据
            backtest_data = {}
            for symbol, data in stock_data.items():
                backtest_data[symbol] = data['Close']
            backtest_data = pd.DataFrame(backtest_data)

            backtester = Backtester(
                strategy=strategy,
                initial_capital=initial_capital,
                commission_rate=commission_rate,
                slippage_rate=slippage_rate
            )

            result = backtester.run(
                data=backtest_data,
                mode=BacktestMode.MULTI_STOCK
            )

            print(f"  [OK] 回测运行成功")
            print(f"       权益曲线长度: {len(result.equity_curve)}")
            print(f"       最终权益: {result.equity_curve.iloc[-1]:,.2f}")
            print(f"       总收益率: {result.performance_metrics.get('总收益率%', 0):.2f}%")
        else:
            print("  [SKIP] 没有数据或策略可测试回测")

    except Exception as e:
        print(f"  [ERROR] 回测测试失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("Web应用逻辑测试完成!")
    print("=" * 80)

    print("\n下一步:")
    print("1. 运行完整的Web应用: streamlit run web_app.py")
    print("2. 在浏览器中打开: http://localhost:8501")
    print("3. 确保启用'使用模拟数据'选项")

except Exception as e:
    print(f"\n[ERROR] 测试过程中出现错误: {e}")
    import traceback
    traceback.print_exc()