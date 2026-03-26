"""数据获取模块 - 支持Akshare和Efinance"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import List, Dict, Optional, Union
from pathlib import Path
import hashlib
from src.utils.config_loader import get_config

# 安装akshare代理补丁，解决访问限制问题
try:
    import akshare_proxy_patch
    akshare_proxy_patch.install_patch("101.201.173.125", "", 50)
    logging.info("akshare代理补丁已安装")
except ImportError:
    logging.warning("akshare-proxy-patch未安装，将使用默认方式获取数据")
except Exception as e:
    logging.warning(f"akshare代理补丁安装失败: {e}")

logger = logging.getLogger(__name__)


class DataFetcher:
    """数据获取器 - 支持多数据源"""

    def __init__(self, cache_enabled: bool = True, data_source: str = "auto"):
        """
        初始化数据获取器

        Args:
            cache_enabled: 是否启用缓存
            data_source: 数据源 ("akshare", "efinance", "auto")
        """
        self.cache_enabled = cache_enabled
        self.config = get_config()
        self.data_source = data_source

        # 创建缓存目录
        if cache_enabled:
            cache_path = Path(self.config.get("cache", {}).get("path", "./data/cache"))
            cache_path.mkdir(parents=True, exist_ok=True)

        # 数据源配置
        self.data_sources_config = self.config.get("data_sources", {})
        self.timeout = 30
        self.retry_count = 3

    def _normalize_symbol(self, symbol: str) -> tuple:
        """
        标准化股票代码，返回(标准代码, 市场标识)

        Args:
            symbol: 股票代码

        Returns:
            (normalized_symbol, market): 标准化代码和市场标识
        """
        symbol = symbol.upper().strip()

        # 处理后缀格式
        if ".SH" in symbol or ".SHE" in symbol:
            code = symbol.split(".")[0]
            return code, "CN"
        elif ".SZ" in symbol:
            code = symbol.split(".")[0]
            return code, "CN"
        elif ".HK" in symbol:
            code = symbol.split(".")[0].zfill(5)
            return code, "HK"
        elif ".US" in symbol:
            code = symbol.split(".")[0]
            return code, "US"

        # 根据代码特征判断市场
        code = symbol.replace(".SZ", "").replace(".SH", "").replace(".HK", "").replace(".US", "")

        # 6位数字 - A股
        if len(code) == 6 and code.isdigit():
            return code, "CN"

        # 5位数字 - 港股
        if len(code) == 5 and code.isdigit():
            return code, "HK"

        # 其他 - 美股
        return code, "US"

    def get_stock_data(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1d",
        adjust: bool = True
    ) -> pd.DataFrame:
        """
        获取股票数据

        Args:
            symbol: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            interval: 数据频率 (1d, 1wk, 1mo)
            adjust: 是否调整价格（除权除息）

        Returns:
            DataFrame包含以下列:
            - Open, High, Low, Close, Volume
        """
        # 转换日期格式
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        # 检查缓存
        cache_key = self._generate_cache_key(symbol, start_date, end_date, interval, adjust)
        cached_data = self._get_from_cache(cache_key)

        if cached_data is not None:
            logger.info(f"从缓存获取数据: {symbol}")
            return cached_data

        # 从数据源获取数据
        data = self._fetch_from_source(symbol, start_date, end_date, interval, adjust)

        # 保存到缓存
        if self.cache_enabled and data is not None and not data.empty:
            self._save_to_cache(cache_key, data)

        return data

    def _fetch_from_source(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str,
        adjust: bool
    ) -> Optional[pd.DataFrame]:
        """从数据源获取数据"""
        normalized_symbol, market = self._normalize_symbol(symbol)
        logger.info(f"获取数据 {symbol} -> {normalized_symbol} ({market})")

        # 非A股只支持Akshare
        if market != "CN":
            return self._fetch_non_cn_stock(symbol, normalized_symbol, market, start_date, end_date, interval)

        # A股数据：根据配置选择数据源
        if self.data_source == "efinance":
            data = self._fetch_with_efinance(normalized_symbol, start_date, end_date, interval, adjust)
            if data is not None and not data.empty:
                return data
        elif self.data_source == "akshare":
            data = self._fetch_with_akshare(normalized_symbol, start_date, end_date, interval, adjust)
            if data is not None and not data.empty:
                return data
        else:  # auto
            # 先尝试Efinance，再尝试Akshare
            for source in ["efinance", "akshare"]:
                try:
                    if source == "efinance":
                        data = self._fetch_with_efinance(normalized_symbol, start_date, end_date, interval, adjust)
                    else:
                        data = self._fetch_with_akshare(normalized_symbol, start_date, end_date, interval, adjust)

                    if data is not None and not data.empty:
                        logger.info(f"使用 {source} 成功获取数据")
                        return data
                except Exception as e:
                    logger.warning(f"{source} 获取失败: {e}")
                    continue

        return None

    def _fetch_with_efinance(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str,
        adjust: bool
    ) -> Optional[pd.DataFrame]:
        """使用Efinance获取A股数据"""
        import efinance as ef

        try:
            # 获取股票历史数据
            df = ef.stock.get_quote_history(
                symbol,
                beg=start_date.strftime("%Y%m%d"),
                end=end_date.strftime("%Y%m%d"),
                klt=1 if interval == "1d" else 5 if interval == "1wk" else 21,  # 日/周/月
                fqt=1 if adjust else 0  # 1=前复权, 0=不复权
            )

            if df is None or df.empty:
                return None

            # 标准化列名
            column_map = {
                "日期": "date",
                "开盘": "Open",
                "最高": "High",
                "最低": "Low",
                "收盘": "Close",
                "成交量": "Volume",
                "成交额": "Amount",
                "振幅": "Amplitude",
                "涨跌幅": "ChangePct",
                "涨跌额": "Change",
                "换手率": "Turnover"
            }
            df = df.rename(columns=column_map)

            # 设置日期索引
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")

            # 确保有必要的列
            required_cols = ["Open", "High", "Low", "Close", "Volume"]
            for col in required_cols:
                if col not in df.columns:
                    return None

            df = df[required_cols].sort_index()
            df = df[~df.index.duplicated(keep='first')]

            return df

        except Exception as e:
            logger.error(f"Efinance获取数据失败 {symbol}: {e}")
            return None

    def _fetch_with_akshare(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str,
        adjust: bool
    ) -> Optional[pd.DataFrame]:
        """使用Akshare获取A股数据"""
        import akshare as ak

        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")

        # 调整类型: qfq=前复权, hfq=后复权, ""=不复权
        adjust_type = "qfq" if adjust else ""

        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period=interval.replace("1d", "daily").replace("1wk", "weekly").replace("1mo", "monthly"),
                start_date=start_str,
                end_date=end_str,
                adjust=adjust_type
            )

            if df is None or df.empty:
                return None

            # 标准化列名
            column_map = {
                "日期": "date",
                "开盘": "Open",
                "最高": "High",
                "最低": "Low",
                "收盘": "Close",
                "成交量": "Volume",
            }
            df = df.rename(columns=column_map)

            # 设置日期索引
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")

            # 确保有必要的列
            required_cols = ["Open", "High", "Low", "Close", "Volume"]
            df = df[[c for c in required_cols if c in df.columns]].sort_index()
            df = df[~df.index.duplicated(keep='first')]

            return df

        except Exception as e:
            logger.error(f"Akshare获取数据失败 {symbol}: {e}")
            return None

    def _fetch_non_cn_stock(
        self,
        symbol: str,
        normalized_symbol: str,
        market: str,
        start_date: datetime,
        end_date: datetime,
        interval: str
    ) -> Optional[pd.DataFrame]:
        """获取非A股数据（港股/美股）"""
        import akshare as ak

        try:
            if market == "HK":
                df = ak.stock_hk_hist(
                    symbol=normalized_symbol,
                    period=interval.replace("1d", "daily").replace("1wk", "weekly").replace("1mo", "monthly"),
                    adjust="qfq"
                )
            else:  # US
                df = ak.stock_us_hist(
                    symbol=normalized_symbol,
                    period=interval.replace("1d", "daily").replace("1wk", "weekly").replace("1mo", "monthly"),
                    adjust="qfq"
                )

            if df is None or df.empty:
                return None

            # 标准化列名
            column_map = {
                "日期": "date",
                "开盘": "Open",
                "最高": "High",
                "最低": "Low",
                "收盘": "Close",
                "成交量": "Volume"
            }
            df = df.rename(columns=column_map)

            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")

            # 过滤日期范围
            df = df[(df.index >= start_date) & (df.index <= end_date)]

            required_cols = ["Open", "High", "Low", "Close", "Volume"]
            df = df[[c for c in required_cols if c in df.columns]].sort_index()

            return df

        except Exception as e:
            logger.error(f"获取{market}股票数据失败 {symbol}: {e}")
            return None

    def get_multiple_stocks(
        self,
        symbols: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1d",
        adjust: bool = True,
        parallel: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        获取多只股票数据

        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            interval: 数据频率
            adjust: 是否调整价格
            parallel: 是否并行获取（暂不支持）

        Returns:
            字典: {symbol: DataFrame}
        """
        data_dict = {}

        for symbol in symbols:
            try:
                data = self.get_stock_data(symbol, start_date, end_date, interval, adjust)
                if data is not None and not data.empty:
                    data_dict[symbol] = data
                else:
                    logger.warning(f"无法获取数据: {symbol}")
            except Exception as e:
                logger.error(f"获取{symbol}数据失败: {e}")

        return data_dict

    def get_index_data(
        self,
        index_symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1d"
    ) -> pd.DataFrame:
        """获取指数数据"""
        import efinance as ef

        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)

        # 指数代码映射
        index_map = {
            "000001": "sh000001",
            "399001": "sz399001",
            "000300": "sh000300",
            "000905": "sh000905",
            "000016": "sh000016",
            "399006": "sz399006",
        }

        code = index_symbol.replace(".SH", "").replace(".SZ", "")
        ef_code = index_map.get(code, code)

        try:
            df = ef.stock.get_quote_history(
                ef_code,
                beg=start_date.strftime("%Y%m%d"),
                end=end_date.strftime("%Y%m%d"),
                klt=1,
                fqt=0
            )

            if df is not None and not df.empty:
                column_map = {"日期": "date", "开盘": "Open", "最高": "High", "最低": "Low", "收盘": "Close", "成交量": "Volume"}
                df = df.rename(columns=column_map)
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.set_index("date")
                return df[["Open", "High", "Low", "Close", "Volume"]].sort_index()

        except Exception as e:
            logger.error(f"获取指数数据失败 {index_symbol}: {e}")

        return pd.DataFrame()

    def get_stock_list(self, market: str = "CN") -> List[str]:
        """获取股票列表"""
        if market == "CN":
            return [
                "600519",  # 贵州茅台
                "000858",  # 五粮液
                "000333",  # 美的集团
                "601318",  # 中国平安
                "600036",  # 招商银行
                "000001",  # 平安银行
                "002415",  # 海康威视
                "300750",  # 宁德时代
                "601888",  # 中国中免
                "002594",  # 比亚迪
            ]
        elif market == "US":
            return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM", "JNJ", "V"]
        elif market == "HK":
            return ["00700", "09988", "01810", "02318", "01299"]
        return []

    def get_realtime_quotes(self, symbols: List[str]) -> pd.DataFrame:
        """获取实时行情"""
        import efinance as ef

        try:
            df = ef.stock.get_realtime_quotes()
            codes = [self._normalize_symbol(s)[0] for s in symbols]
            result = df[df["股票代码"].isin(codes)]

            if not result.empty:
                return result[["股票代码", "股票名称", "最新价", "涨跌幅", "成交量", "成交额", "最高", "最低", "今开"]]

        except Exception as e:
            logger.error(f"获取实时行情失败: {e}")

        return pd.DataFrame()

    def get_fundamental_data(self, symbol: str) -> Dict:
        """获取基本面数据"""
        # 简化版本，返回空字典
        return {}

    def _generate_cache_key(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        interval: str,
        adjust: bool
    ) -> str:
        """生成缓存键"""
        key_str = f"{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}_{interval}_{adjust}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """从缓存获取数据"""
        if not self.cache_enabled:
            return None

        cache_path = Path(self.config.get("cache", {}).get("path", "./data/cache"))
        cache_file = cache_path / f"{cache_key}.parquet"

        if cache_file.exists():
            try:
                cache_age = time.time() - cache_file.stat().st_mtime
                ttl_days = self.config.get("cache", {}).get("ttl_days", 7)
                ttl_seconds = ttl_days * 24 * 3600

                if cache_age < ttl_seconds:
                    data = pd.read_parquet(cache_file)
                    logger.debug(f"从缓存读取数据: {cache_file}")
                    return data
                else:
                    cache_file.unlink()
            except Exception as e:
                logger.warning(f"读取缓存失败: {e}")

        return None

    def _save_to_cache(self, cache_key: str, data: pd.DataFrame):
        """保存数据到缓存"""
        if not self.cache_enabled:
            return

        try:
            cache_path = Path(self.config.get("cache", {}).get("path", "./data/cache"))
            cache_file = cache_path / f"{cache_key}.parquet"
            cache_path.mkdir(parents=True, exist_ok=True)
            data.to_parquet(cache_file)
            logger.debug(f"数据保存到缓存: {cache_file}")
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")

    def clear_cache(self, older_than_days: int = None):
        """清理缓存"""
        if not self.cache_enabled:
            return

        cache_path = Path(self.config.get("cache", {}).get("path", "./data/cache"))
        if not cache_path.exists():
            return

        current_time = time.time()
        deleted_count = 0

        for cache_file in cache_path.glob("*.parquet"):
            try:
                if older_than_days is None:
                    cache_file.unlink()
                    deleted_count += 1
                else:
                    file_age = current_time - cache_file.stat().st_mtime
                    if file_age > older_than_days * 24 * 3600:
                        cache_file.unlink()
                        deleted_count += 1
            except Exception as e:
                logger.warning(f"删除缓存文件失败 {cache_file}: {e}")

        logger.info(f"清理了 {deleted_count} 个缓存文件")


# 测试函数
def test_data_fetcher():
    """测试数据获取器"""
    import logging
    logging.basicConfig(level=logging.INFO)

    fetcher = DataFetcher()

    print("测试获取A股数据...")
    data = fetcher.get_stock_data("600519", "2024-01-01", "2024-03-01")
    if data is not None and not data.empty:
        print(f"贵州茅台数据形状: {data.shape}")
        print(f"数据列: {data.columns.tolist()}")
        print(f"前5行:\n{data.head()}")
    else:
        print("获取数据失败")


if __name__ == "__main__":
    test_data_fetcher()
