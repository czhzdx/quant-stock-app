"""股票输入处理模块"""
import os
import re
import logging
from typing import List, Optional, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from src.utils.config_loader import get_config

logger = logging.getLogger(__name__)


class StockInputSource(Enum):
    """股票输入来源"""
    CLI = "cli"              # 命令行参数
    ENV_VAR = "env_var"      # 环境变量
    CONFIG_FILE = "config"   # 配置文件
    WORKFLOW = "workflow"    # GitHub Actions workflow_dispatch
    DEFAULT = "default"      # 默认值


@dataclass
class StockInput:
    """股票输入结果"""
    symbols: List[str]
    source: StockInputSource
    raw_input: str


def normalize_symbol(symbol: str) -> str:
    """
    标准化股票代码

    支持格式:
    - 600519 -> 600519
    - 600519.SH -> 600519
    - 600519.SHE -> 600519
    - sh600519 -> 600519
    - 00700.HK -> 00700
    - AAPL.US -> AAPL
    """
    symbol = symbol.strip().upper()

    # 移除前缀
    if symbol.startswith("SH") or symbol.startswith("SZ"):
        symbol = symbol[2:]
    elif symbol.startswith("HK"):
        symbol = symbol[2:].zfill(5)  # 港股补零

    # 移除后缀
    for suffix in [".SH", ".SHE", ".SZ", ".HK", ".US", ".BJ"]:
        symbol = symbol.replace(suffix, "")

    # 验证格式
    if not symbol:
        return None

    return symbol


def parse_stock_list(input_str: str, separator: str = None) -> List[str]:
    """
    解析股票列表字符串

    支持格式:
    - "600519,000858,000333"  (逗号分隔)
    - "600519 000858 000333"  (空格分隔)
    - "600519\n000858\n000333" (换行分隔)
    - "600519, 000858, 000333" (带空格)

    Args:
        input_str: 输入字符串
        separator: 指定分隔符（默认自动检测）

    Returns:
        标准化后的股票代码列表
    """
    if not input_str:
        return []

    # 自动检测分隔符
    if separator is None:
        if "," in input_str:
            separator = ","
        elif "\n" in input_str:
            separator = "\n"
        elif ";" in input_str:
            separator = ";"
        else:
            separator = " "

    # 分割并标准化
    symbols = []
    for s in input_str.split(separator):
        s = s.strip()
        if s:
            normalized = normalize_symbol(s)
            if normalized:
                symbols.append(normalized)

    return symbols


def get_stocks(
    cli_stocks: Optional[Union[str, List[str]]] = None,
    env_var: str = "STOCK_LIST",
    workflow_input: str = None,
    config_key: str = "stock_analysis.stocks.default_list"
) -> StockInput:
    """
    获取股票列表（按优先级）

    优先级: CLI > 工作流输入 > 环境变量 > 配置文件 > 默认值

    Args:
        cli_stocks: 命令行参数传入的股票列表
        env_var: 环境变量名称
        workflow_input: GitHub Actions workflow_dispatch 输入
        config_key: 配置文件中的key

    Returns:
        StockInput: 包含股票列表和来源信息
    """
    # 1. 命令行参数（最高优先级）
    if cli_stocks:
        if isinstance(cli_stocks, str):
            symbols = parse_stock_list(cli_stocks)
        else:
            symbols = [normalize_symbol(s) for s in cli_stocks if s]
            symbols = [s for s in symbols if s]

        if symbols:
            logger.info(f"从CLI参数获取股票列表: {symbols}")
            return StockInput(
                symbols=symbols,
                source=StockInputSource.CLI,
                raw_input=str(cli_stocks)
            )

    # 2. GitHub Actions workflow_dispatch 输入
    if workflow_input:
        symbols = parse_stock_list(workflow_input)
        if symbols:
            logger.info(f"从workflow_dispatch获取股票列表: {symbols}")
            return StockInput(
                symbols=symbols,
                source=StockInputSource.WORKFLOW,
                raw_input=workflow_input
            )

    # 3. 环境变量
    env_value = os.environ.get(env_var, "").strip()
    if env_value:
        symbols = parse_stock_list(env_value)
        if symbols:
            logger.info(f"从环境变量{env_var}获取股票列表: {symbols}")
            return StockInput(
                symbols=symbols,
                source=StockInputSource.ENV_VAR,
                raw_input=env_value
            )

    # 4. 配置文件
    try:
        config = get_config()
        config_stocks = config.get(config_key, [])
        if config_stocks:
            symbols = [normalize_symbol(s) for s in config_stocks if s]
            symbols = [s for s in symbols if s]
            if symbols:
                logger.info(f"从配置文件获取股票列表: {symbols}")
                return StockInput(
                    symbols=symbols,
                    source=StockInputSource.CONFIG_FILE,
                    raw_input=str(config_stocks)
                )
    except Exception as e:
        logger.warning(f"读取配置文件失败: {e}")

    # 5. 默认值
    default_stocks = ["600519", "000858", "000333"]  # 茅台、五粮液、美的
    logger.info(f"使用默认股票列表: {default_stocks}")
    return StockInput(
        symbols=default_stocks,
        source=StockInputSource.DEFAULT,
        raw_input=""
    )


def get_stocks_from_file(file_path: str) -> List[str]:
    """
    从文件读取股票列表

    文件格式:
    - 每行一个股票代码
    - 支持注释 (#开头)
    - 支持逗号分隔的行

    Args:
        file_path: 文件路径

    Returns:
        股票代码列表
    """
    path = Path(file_path)
    if not path.exists():
        logger.error(f"股票列表文件不存在: {file_path}")
        return []

    symbols = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # 跳过空行和注释
            if not line or line.startswith("#"):
                continue
            # 解析行
            line_symbols = parse_stock_list(line)
            symbols.extend(line_symbols)

    logger.info(f"从文件 {file_path} 读取到 {len(symbols)} 只股票")
    return symbols


def validate_stocks(symbols: List[str], market: str = None) -> List[str]:
    """
    验证股票代码格式

    Args:
        symbols: 股票代码列表
        market: 市场限制 ("CN", "HK", "US")

    Returns:
        有效的股票代码列表
    """
    valid_symbols = []

    for symbol in symbols:
        # 基本格式检查
        if not symbol or not isinstance(symbol, str):
            continue

        # A股: 6位数字
        if len(symbol) == 6 and symbol.isdigit():
            if market and market != "CN":
                continue
            valid_symbols.append(symbol)

        # 港股: 5位数字
        elif len(symbol) == 5 and symbol.isdigit():
            if market and market != "HK":
                continue
            valid_symbols.append(symbol)

        # 美股: 字母
        elif symbol.isalpha():
            if market and market != "US":
                continue
            valid_symbols.append(symbol.upper())

        else:
            logger.warning(f"无效的股票代码格式: {symbol}")

    return valid_symbols


# 便捷函数
def get_stock_list(
    stocks: Optional[Union[str, List[str]]] = None,
    file: str = None,
    env: str = "STOCK_LIST",
    validate: bool = True
) -> List[str]:
    """
    获取股票列表的便捷函数

    Args:
        stocks: 直接传入的股票列表
        file: 股票列表文件路径
        env: 环境变量名
        validate: 是否验证格式

    Returns:
        股票代码列表
    """
    if stocks:
        if isinstance(stocks, str):
            result = parse_stock_list(stocks)
        else:
            result = [normalize_symbol(s) for s in stocks if s]
            result = [s for s in result if s]
    elif file:
        result = get_stocks_from_file(file)
    else:
        result = get_stocks(env_var=env).symbols

    if validate:
        result = validate_stocks(result)

    return result
