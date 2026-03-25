"""量化选股软件核心模块"""

from src.data import DataFetcher
from src.factors import FactorCalculator, FactorType
from src.strategies import BaseStrategy, SignalType, MomentumStrategy, DualMomentumStrategy, MeanReversionStrategy, MultiFactorStrategy
from src.backtest import Backtester, BacktestMode, BacktestResult
from src.visualization import Plotter
from src.utils import get_config, get_config_loader

__all__ = [
    'DataFetcher',
    'FactorCalculator',
    'FactorType',
    'BaseStrategy',
    'SignalType',
    'MomentumStrategy',
    'DualMomentumStrategy',
    'MeanReversionStrategy',
    'MultiFactorStrategy',
    'Backtester',
    'BacktestMode',
    'BacktestResult',
    'Plotter',
    'get_config',
    'get_config_loader',
]
