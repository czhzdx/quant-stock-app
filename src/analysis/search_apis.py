"""搜索增强API模块 - Tavily/SerpAPI/博查集成"""
import os
import json
import logging
import aiohttp
import asyncio
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod

from src.utils.config_loader import get_config
from src.utils.fallback_chain import FallbackChain

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """搜索结果"""
    title: str
    url: str
    snippet: str
    source: str
    published_date: str = None

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source": self.source,
            "published_date": self.published_date
        }


@dataclass
class SearchInfo:
    """股票搜索信息"""
    symbol: str
    news: List[SearchResult] = field(default_factory=list)
    announcements: List[SearchResult] = field(default_factory=list)
    research_reports: List[SearchResult] = field(default_factory=list)
    industry_news: List[SearchResult] = field(default_factory=list)
    search_time: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "news": [n.to_dict() for n in self.news],
            "announcements": [a.to_dict() for a in self.announcements],
            "research_reports": [r.to_dict() for r in self.research_reports],
            "industry_news": [i.to_dict() for i in self.industry_news],
            "search_time": self.search_time.isoformat()
        }

    def get_all_text(self) -> str:
        """获取所有搜索结果的文本"""
        texts = []
        for item in self.news + self.announcements + self.research_reports:
            texts.append(f"标题: {item.title}\n内容: {item.snippet}")
        return "\n\n".join(texts)


class BaseSearchProvider(ABC):
    """搜索提供者基类"""

    def __init__(self, api_key: str = None, **kwargs):
        self.api_key = api_key
        self.config = kwargs

    @abstractmethod
    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """执行搜索"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """提供者名称"""
        pass


class TavilyProvider(BaseSearchProvider):
    """Tavily搜索提供者"""

    BASE_URL = "https://api.tavily.com/search"

    @property
    def name(self) -> str:
        return "tavily"

    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """使用Tavily搜索"""
        if not self.api_key:
            raise ValueError("Tavily API key not configured")

        async with aiohttp.ClientSession() as session:
            payload = {
                "api_key": self.api_key,
                "query": query,
                "search_depth": "basic",
                "max_results": num_results,
                "include_raw_content": False
            }

            async with session.post(self.BASE_URL, json=payload) as response:
                if response.status != 200:
                    raise Exception(f"Tavily API error: {response.status}")

                data = await response.json()

            results = []
            for item in data.get("results", []):
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                    source="tavily",
                    published_date=item.get("published_date")
                ))

            return results


class SerpAPIProvider(BaseSearchProvider):
    """SerpAPI搜索提供者"""

    BASE_URL = "https://serpapi.com/search"

    @property
    def name(self) -> str:
        return "serpapi"

    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """使用SerpAPI搜索"""
        if not self.api_key:
            raise ValueError("SerpAPI key not configured")

        async with aiohttp.ClientSession() as session:
            params = {
                "q": query,
                "api_key": self.api_key,
                "num": num_results
            }

            async with session.get(self.BASE_URL, params=params) as response:
                if response.status != 200:
                    raise Exception(f"SerpAPI error: {response.status}")

                data = await response.json()

            results = []
            for item in data.get("organic_results", []):
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("link", ""),
                    snippet=item.get("snippet", ""),
                    source="serpapi"
                ))

            return results


class BochaProvider(BaseSearchProvider):
    """博查搜索提供者"""

    BASE_URL = "https://api.bocha.io/v1/search"

    @property
    def name(self) -> str:
        return "bocha"

    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """使用博查搜索"""
        if not self.api_key:
            raise ValueError("Bocha API key not configured")

        async with aiohttp.ClientSession() as session:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "query": query,
                "count": num_results
            }

            async with session.post(self.BASE_URL, json=payload, headers=headers) as response:
                if response.status != 200:
                    raise Exception(f"Bocha API error: {response.status}")

                data = await response.json()

            results = []
            for item in data.get("data", {}).get("results", []):
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", item.get("snippet", "")),
                    source="bocha"
                ))

            return results


class SearchAPIManager:
    """搜索API管理器"""

    def __init__(self):
        """初始化搜索管理器"""
        self.config = get_config()
        self.search_config = self.config.get("stock_analysis", {}).get("search_apis", {})

        # 初始化提供者
        self.providers: Dict[str, BaseSearchProvider] = {}
        self._init_providers()

        # 设置降级顺序
        self.primary = self.search_config.get("primary", "tavily")
        self.fallback = self.search_config.get("fallback", ["serpapi", "bocha"])

    def _init_providers(self):
        """初始化搜索提供者"""
        # Tavily
        tavily_key = self._get_api_key("tavily")
        if tavily_key:
            self.providers["tavily"] = TavilyProvider(api_key=tavily_key)

        # SerpAPI
        serpapi_key = self._get_api_key("serpapi")
        if serpapi_key:
            self.providers["serpapi"] = SerpAPIProvider(api_key=serpapi_key)

        # Bocha
        bocha_key = self._get_api_key("bocha")
        if bocha_key:
            self.providers["bocha"] = BochaProvider(api_key=bocha_key)

    def _get_api_key(self, provider: str) -> Optional[str]:
        """获取API密钥（支持环境变量）"""
        provider_config = self.search_config.get(provider, {})
        api_key = provider_config.get("api_key", "")

        # 处理环境变量引用
        if api_key and api_key.startswith("${") and api_key.endswith("}"):
            env_var = api_key[2:-1]
            api_key = os.environ.get(env_var, "")

        return api_key if api_key else None

    async def search(
        self,
        query: str,
        num_results: int = 10,
        provider: str = None
    ) -> List[SearchResult]:
        """
        执行搜索（支持降级）

        Args:
            query: 搜索查询
            num_results: 结果数量
            provider: 指定提供者（None则使用降级链）

        Returns:
            搜索结果列表
        """
        if provider and provider in self.providers:
            return await self.providers[provider].search(query, num_results)

        # 使用降级链
        order = [self.primary] + self.fallback
        last_error = None

        for prov_name in order:
            if prov_name not in self.providers:
                continue

            try:
                logger.info(f"尝试使用 {prov_name} 搜索")
                results = await self.providers[prov_name].search(query, num_results)
                if results:
                    logger.info(f"{prov_name} 搜索成功，返回 {len(results)} 条结果")
                    return results
            except Exception as e:
                logger.warning(f"{prov_name} 搜索失败: {e}")
                last_error = e
                continue

        logger.error("所有搜索提供者都失败")
        return []

    async def search_stock_info(
        self,
        symbol: str,
        stock_name: str = None,
        include_news: bool = True,
        include_announcements: bool = True,
        include_research: bool = True
    ) -> SearchInfo:
        """
        搜索股票相关信息

        Args:
            symbol: 股票代码
            stock_name: 股票名称
            include_news: 是否搜索新闻
            include_announcements: 是否搜索公告
            include_research: 是否搜索研报

        Returns:
            SearchInfo: 搜索信息
        """
        logger.info(f"搜索股票信息: {symbol}")

        search_info = SearchInfo(symbol=symbol)

        # 构建搜索关键词
        keywords = [symbol]
        if stock_name:
            keywords.append(stock_name)

        search_keyword = " ".join(keywords)

        # 并行搜索
        tasks = []

        if include_news:
            tasks.append(("news", f"{search_keyword} 股票新闻"))

        if include_announcements:
            tasks.append(("announcements", f"{search_keyword} 公告"))

        if include_research:
            tasks.append(("research", f"{search_keyword} 研报 分析"))

        # 执行搜索
        for category, query in tasks:
            try:
                results = await self.search(query, num_results=5)
                if category == "news":
                    search_info.news = results
                elif category == "announcements":
                    search_info.announcements = results
                elif category == "research":
                    search_info.research_reports = results
            except Exception as e:
                logger.error(f"搜索 {category} 失败: {e}")

        return search_info

    def search_sync(
        self,
        query: str,
        num_results: int = 10
    ) -> List[SearchResult]:
        """同步搜索"""
        return asyncio.run(self.search(query, num_results))
