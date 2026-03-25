"""增强数据获取模块 - 支持历史/实时/筹码数据"""
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from src.data.data_fetcher import DataFetcher
from src.utils.config_loader import get_config
from src.utils.fallback_chain import FallbackChain

logger = logging.getLogger(__name__)


@dataclass
class RealTimeQuote:
    """实时行情数据"""
    symbol: str
    name: str
    price: float
    change: float
    change_pct: float
    open: float
    high: float
    low: float
    volume: int
    amount: float
    turnover_rate: float = 0.0
    timestamp: datetime = None

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "name": self.name,
            "price": self.price,
            "change": self.change,
            "change_pct": self.change_pct,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "volume": self.volume,
            "amount": self.amount,
            "turnover_rate": self.turnover_rate,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }


@dataclass
class ChipDistribution:
    """筹码分布数据"""
    symbol: str
    date: datetime
    # 筹码集中度
    concentration_ratio_90: float = 0.0  # 90%筹码集中度
    concentration_ratio_70: float = 0.0  # 70%筹码集中度
    # 获利盘比例
    profit_ratio: float = 0.0
    # 平均成本
    avg_cost: float = 0.0
    # 筹码峰
    chip_peaks: List[Dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "date": self.date.isoformat(),
            "concentration_ratio_90": self.concentration_ratio_90,
            "concentration_ratio_70": self.concentration_ratio_70,
            "profit_ratio": self.profit_ratio,
            "avg_cost": self.avg_cost,
            "chip_peaks": self.chip_peaks
        }


@dataclass
class InstitutionalHolding:
    """机构持仓数据"""
    symbol: str
    report_date: datetime
    institution_name: str
    shares: int
    shares_pct: float
    change: int
    change_pct: float

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "report_date": self.report_date.isoformat(),
            "institution_name": self.institution_name,
            "shares": self.shares,
            "shares_pct": self.shares_pct,
            "change": self.change,
            "change_pct": self.change_pct
        }


@dataclass
class StockData:
    """完整股票数据"""
    symbol: str
    # 历史K线
    historical: pd.DataFrame = None
    # 实时行情
    realtime: RealTimeQuote = None
    # 筹码分布
    chip_distribution: ChipDistribution = None
    # 机构持仓
    institutional_holdings: List[InstitutionalHolding] = field(default_factory=list)
    # 龙虎榜
    long_short_board: pd.DataFrame = None
    # 获取时间
    fetch_time: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "historical": self.historical.to_dict() if self.historical is not None else None,
            "realtime": self.realtime.to_dict() if self.realtime else None,
            "chip_distribution": self.chip_distribution.to_dict() if self.chip_distribution else None,
            "institutional_holdings": [h.to_dict() for h in self.institutional_holdings],
            "fetch_time": self.fetch_time.isoformat()
        }


class EnhancedDataFetcher:
    """增强数据获取器"""

    def __init__(self, cache_enabled: bool = True):
        """
        初始化增强数据获取器

        Args:
            cache_enabled: 是否启用缓存
        """
        self.config = get_config()
        self.cache_enabled = cache_enabled

        # 基础数据获取器
        self.base_fetcher = DataFetcher(cache_enabled=cache_enabled)

        # 获取配置
        self.analysis_config = self.config.get("stock_analysis", {})
        self.data_config = self.analysis_config.get("data_fetching", {})

    def get_historical_data(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        获取历史K线数据

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            interval: 周期 (1d, 1wk, 1mo)

        Returns:
            DataFrame: OHLCV数据
        """
        logger.info(f"获取历史数据: {symbol}")
        return self.base_fetcher.get_stock_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            adjust=True
        )

    def get_realtime_quote(self, symbol: str) -> Optional[RealTimeQuote]:
        """
        获取实时行情

        Args:
            symbol: 股票代码

        Returns:
            RealTimeQuote: 实时行情数据
        """
        logger.info(f"获取实时行情: {symbol}")

        # 判断市场
        if len(symbol) == 6 and symbol.isdigit():
            market = "CN"
        elif len(symbol) == 5 and symbol.isdigit():
            market = "HK"
        else:
            market = "US"

        try:
            if market == "CN":
                return self._get_cn_realtime(symbol)
            elif market == "HK":
                return self._get_hk_realtime(symbol)
            else:
                return self._get_us_realtime(symbol)
        except Exception as e:
            logger.error(f"获取实时行情失败 {symbol}: {e}")
            return None

    def _get_cn_realtime(self, symbol: str) -> Optional[RealTimeQuote]:
        """获取A股实时行情"""
        try:
            import akshare as ak

            # 获取实时行情
            df = ak.stock_zh_a_spot_em()
            row = df[df["代码"] == symbol]

            if row.empty:
                logger.warning(f"未找到股票: {symbol}")
                return None

            row = row.iloc[0]

            return RealTimeQuote(
                symbol=symbol,
                name=row.get("名称", ""),
                price=float(row.get("最新价", 0)),
                change=float(row.get("涨跌额", 0)),
                change_pct=float(row.get("涨跌幅", 0)),
                open=float(row.get("今开", 0)),
                high=float(row.get("最高", 0)),
                low=float(row.get("最低", 0)),
                volume=int(row.get("成交量", 0)),
                amount=float(row.get("成交额", 0)),
                turnover_rate=float(row.get("换手率", 0)),
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"A股实时行情获取失败: {e}")
            return None

    def _get_hk_realtime(self, symbol: str) -> Optional[RealTimeQuote]:
        """获取港股实时行情"""
        try:
            import akshare as ak

            df = ak.stock_hk_spot_em()
            row = df[df["代码"] == symbol]

            if row.empty:
                logger.warning(f"未找到港股: {symbol}")
                return None

            row = row.iloc[0]

            return RealTimeQuote(
                symbol=symbol,
                name=row.get("名称", ""),
                price=float(row.get("最新价", 0)),
                change=float(row.get("涨跌额", 0)),
                change_pct=float(row.get("涨跌幅", 0)),
                open=float(row.get("今开", 0)),
                high=float(row.get("最高", 0)),
                low=float(row.get("最低", 0)),
                volume=int(row.get("成交量", 0)),
                amount=float(row.get("成交额", 0)),
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"港股实时行情获取失败: {e}")
            return None

    def _get_us_realtime(self, symbol: str) -> Optional[RealTimeQuote]:
        """获取美股实时行情"""
        try:
            import akshare as ak

            df = ak.stock_us_spot_em()
            row = df[df["代码"] == symbol]

            if row.empty:
                logger.warning(f"未找到美股: {symbol}")
                return None

            row = row.iloc[0]

            return RealTimeQuote(
                symbol=symbol,
                name=row.get("名称", ""),
                price=float(row.get("最新价", 0)),
                change=float(row.get("涨跌额", 0)),
                change_pct=float(row.get("涨跌幅", 0)),
                open=float(row.get("今开", 0)),
                high=float(row.get("最高", 0)),
                low=float(row.get("最低", 0)),
                volume=int(row.get("成交量", 0)),
                amount=float(row.get("成交额", 0)),
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.error(f"美股实时行情获取失败: {e}")
            return None

    def get_chip_distribution(self, symbol: str) -> Optional[ChipDistribution]:
        """
        获取筹码分布数据

        Args:
            symbol: 股票代码

        Returns:
            ChipDistribution: 筹码分布数据
        """
        logger.info(f"获取筹码分布: {symbol}")

        try:
            import akshare as ak

            # 获取筹码分布
            df = ak.stock_em_ycfp(symbol)

            if df is None or df.empty:
                logger.warning(f"未获取到筹码数据: {symbol}")
                return None

            # 解析数据
            latest = df.iloc[-1]

            return ChipDistribution(
                symbol=symbol,
                date=datetime.now(),
                profit_ratio=float(latest.get("获利比例", 0)),
                avg_cost=float(latest.get("平均成本", 0)),
                concentration_ratio_90=float(latest.get("90%成本", 0)),
                concentration_ratio_70=float(latest.get("70%成本", 0))
            )
        except Exception as e:
            logger.error(f"筹码分布获取失败: {e}")
            # 返回默认数据
            return ChipDistribution(
                symbol=symbol,
                date=datetime.now()
            )

    def get_institutional_holdings(
        self,
        symbol: str
    ) -> List[InstitutionalHolding]:
        """
        获取机构持仓数据

        Args:
            symbol: 股票代码

        Returns:
            List[InstitutionalHolding]: 机构持仓列表
        """
        logger.info(f"获取机构持仓: {symbol}")
        holdings = []

        try:
            import akshare as ak

            # 获取机构持仓
            df = ak.stock_institute_hold_detail(symbol=symbol)

            if df is None or df.empty:
                logger.warning(f"未获取到机构持仓数据: {symbol}")
                return holdings

            for _, row in df.iterrows():
                holding = InstitutionalHolding(
                    symbol=symbol,
                    report_date=pd.to_datetime(row.get("报告期", datetime.now())),
                    institution_name=row.get("机构名称", ""),
                    shares=int(row.get("持股数量", 0)),
                    shares_pct=float(row.get("持股比例", 0)),
                    change=int(row.get("增减", 0)),
                    change_pct=float(row.get("变动比例", 0))
                )
                holdings.append(holding)

        except Exception as e:
            logger.error(f"机构持仓获取失败: {e}")

        return holdings

    def get_long_short_board(self, symbol: str) -> pd.DataFrame:
        """
        获取龙虎榜数据

        Args:
            symbol: 股票代码

        Returns:
            DataFrame: 龙虎榜数据
        """
        logger.info(f"获取龙虎榜: {symbol}")

        try:
            import akshare as ak

            df = ak.stock_lhb_detail_em(symbol=symbol)

            if df is not None and not df.empty:
                return df

        except Exception as e:
            logger.error(f"龙虎榜获取失败: {e}")

        return pd.DataFrame()

    def get_full_stock_data(
        self,
        symbol: str,
        start_date: Union[str, datetime] = None,
        end_date: Union[str, datetime] = None,
        include_realtime: bool = True,
        include_chip: bool = True,
        include_institutional: bool = False
    ) -> StockData:
        """
        获取完整股票数据

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            include_realtime: 是否包含实时行情
            include_chip: 是否包含筹码分布
            include_institutional: 是否包含机构持仓

        Returns:
            StockData: 完整股票数据
        """
        logger.info(f"获取完整数据: {symbol}")

        # 默认日期范围
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now()

        stock_data = StockData(symbol=symbol)

        # 1. 历史数据
        try:
            stock_data.historical = self.get_historical_data(
                symbol, start_date, end_date
            )
        except Exception as e:
            logger.error(f"历史数据获取失败: {e}")

        # 2. 实时行情
        if include_realtime:
            try:
                stock_data.realtime = self.get_realtime_quote(symbol)
            except Exception as e:
                logger.error(f"实时行情获取失败: {e}")

        # 3. 筹码分布
        if include_chip:
            try:
                stock_data.chip_distribution = self.get_chip_distribution(symbol)
            except Exception as e:
                logger.error(f"筹码分布获取失败: {e}")

        # 4. 机构持仓
        if include_institutional:
            try:
                stock_data.institutional_holdings = self.get_institutional_holdings(symbol)
            except Exception as e:
                logger.error(f"机构持仓获取失败: {e}")

        return stock_data

    def get_multiple_stocks_data(
        self,
        symbols: List[str],
        start_date: Union[str, datetime] = None,
        end_date: Union[str, datetime] = None,
        **kwargs
    ) -> Dict[str, StockData]:
        """
        获取多只股票数据

        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            **kwargs: 其他参数传递给get_full_stock_data

        Returns:
            Dict[str, StockData]: 股票数据字典
        """
        result = {}

        for symbol in symbols:
            try:
                data = self.get_full_stock_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    **kwargs
                )
                result[symbol] = data
            except Exception as e:
                logger.error(f"获取 {symbol} 数据失败: {e}")

        return result
