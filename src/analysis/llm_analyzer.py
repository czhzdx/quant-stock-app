"""LLM分析器模块 - OpenAI/Claude/DeepSeek集成"""
import os
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod

from src.utils.config_loader import get_config
from src.analysis.data_enhanced import StockData, RealTimeQuote, ChipDistribution
from src.analysis.search_apis import SearchInfo

logger = logging.getLogger(__name__)


# 分析提示模板
ANALYSIS_PROMPT_TEMPLATE = """你是一位专业的股票分析师。请根据以下信息对股票进行全面分析。

## 股票基本信息
- 股票代码: {symbol}
- 股票名称: {stock_name}

## 市场数据
{market_data}

## 技术指标
{technical_indicators}

## 筹码分布
{chip_data}

## 相关新闻与公告
{search_info}

## 分析要求

请提供以下内容：

### 1. 基本面分析
- 公司业务概况
- 行业地位
- 主要风险

### 2. 技术面分析
- 趋势判断（上涨/下跌/震荡）
- 支撑位和压力位
- 成交量分析

### 3. 筹码分析
- 筹码集中度
- 获利盘情况
- 主力动向判断

### 4. 消息面分析
- 近期重要新闻解读
- 公告影响评估

### 5. 投资建议
- 综合评级（强烈推荐/推荐/中性/不推荐/强烈不推荐）
- 目标价位（如有）
- 风险提示

### 6. 操作建议
- 建仓策略
- 止损位
- 止盈位

请用专业的语言进行分析，结论要有理有据。
"""


@dataclass
class AnalysisReport:
    """分析报告"""
    symbol: str
    stock_name: str = ""
    analysis_time: datetime = field(default_factory=datetime.now)

    # 分析内容
    fundamental_analysis: str = ""
    technical_analysis: str = ""
    chip_analysis: str = ""
    news_analysis: str = ""

    # 评级与建议
    rating: str = "中性"  # 强烈推荐/推荐/中性/不推荐/强烈不推荐
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    # 风险提示
    risks: List[str] = field(default_factory=list)

    # 操作建议
    operation_advice: str = ""

    # 原始LLM响应
    raw_response: str = ""

    # 使用的模型
    model_used: str = ""

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "stock_name": self.stock_name,
            "analysis_time": self.analysis_time.isoformat(),
            "fundamental_analysis": self.fundamental_analysis,
            "technical_analysis": self.technical_analysis,
            "chip_analysis": self.chip_analysis,
            "news_analysis": self.news_analysis,
            "rating": self.rating,
            "target_price": self.target_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "risks": self.risks,
            "operation_advice": self.operation_advice,
            "model_used": self.model_used
        }


class BaseLLMProvider(ABC):
    """LLM提供者基类"""

    def __init__(self, api_key: str = None, model: str = None, **kwargs):
        self.api_key = api_key
        self.model = model
        self.config = kwargs

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """生成文本"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """提供者名称"""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI提供者"""

    def __init__(self, api_key: str = None, model: str = "gpt-4o", **kwargs):
        super().__init__(api_key, model, **kwargs)
        self.base_url = kwargs.get("base_url", "https://api.openai.com/v1")

    @property
    def name(self) -> str:
        return "openai"

    async def generate(self, prompt: str) -> str:
        """使用OpenAI生成"""
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )

            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一位专业的股票分析师，擅长基本面分析、技术分析和筹码分析。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000,
                temperature=0.7
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"OpenAI生成失败: {e}")
            raise


class AnthropicProvider(BaseLLMProvider):
    """Anthropic提供者"""

    def __init__(self, api_key: str = None, model: str = "claude-sonnet-4-6", **kwargs):
        super().__init__(api_key, model, **kwargs)

    @property
    def name(self) -> str:
        return "anthropic"

    async def generate(self, prompt: str) -> str:
        """使用Anthropic生成"""
        try:
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic(api_key=self.api_key)

            response = await client.messages.create(
                model=self.model,
                max_tokens=4000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            return response.content[0].text

        except Exception as e:
            logger.error(f"Anthropic生成失败: {e}")
            raise


class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek提供者"""

    def __init__(self, api_key: str = None, model: str = "deepseek-chat", **kwargs):
        super().__init__(api_key, model, **kwargs)
        self.base_url = kwargs.get("base_url", "https://api.deepseek.com/v1")

    @property
    def name(self) -> str:
        return "deepseek"

    async def generate(self, prompt: str) -> str:
        """使用DeepSeek生成"""
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )

            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一位专业的股票分析师，擅长基本面分析、技术分析和筹码分析。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=4000,
                temperature=0.7
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"DeepSeek生成失败: {e}")
            raise


class LLMAnalyzer:
    """LLM分析器"""

    def __init__(self):
        """初始化分析器"""
        self.config = get_config()
        self.llm_config = self.config.get("stock_analysis", {}).get("llm", {})

        # 初始化提供者
        self.providers: Dict[str, BaseLLMProvider] = {}
        self._init_providers()

        # 设置降级顺序
        self.primary = self.llm_config.get("primary", "openai")
        self.fallback = self.llm_config.get("fallback", ["anthropic", "deepseek"])

    def _init_providers(self):
        """初始化LLM提供者"""
        # OpenAI
        openai_config = self.llm_config.get("openai", {})
        openai_key = self._get_api_key("openai")
        if openai_key:
            self.providers["openai"] = OpenAIProvider(
                api_key=openai_key,
                model=openai_config.get("model", "gpt-4o")
            )

        # Anthropic
        anthropic_config = self.llm_config.get("anthropic", {})
        anthropic_key = self._get_api_key("anthropic")
        if anthropic_key:
            self.providers["anthropic"] = AnthropicProvider(
                api_key=anthropic_key,
                model=anthropic_config.get("model", "claude-sonnet-4-6")
            )

        # DeepSeek
        deepseek_config = self.llm_config.get("deepseek", {})
        deepseek_key = self._get_api_key("deepseek")
        if deepseek_key:
            self.providers["deepseek"] = DeepSeekProvider(
                api_key=deepseek_key,
                model=deepseek_config.get("model", "deepseek-chat")
            )

    def _get_api_key(self, provider: str) -> Optional[str]:
        """获取API密钥"""
        provider_config = self.llm_config.get(provider, {})
        api_key = provider_config.get("api_key", "")

        # 处理环境变量引用
        if api_key and api_key.startswith("${") and api_key.endswith("}"):
            env_var = api_key[2:-1]
            api_key = os.environ.get(env_var, "")

        return api_key if api_key else None

    async def generate(
        self,
        prompt: str,
        provider: str = None
    ) -> tuple:
        """
        生成文本（支持降级）

        Args:
            prompt: 提示文本
            provider: 指定提供者

        Returns:
            (响应文本, 使用的提供者名称)
        """
        if provider and provider in self.providers:
            response = await self.providers[provider].generate(prompt)
            return response, provider

        # 使用降级链
        order = [self.primary] + self.fallback
        last_error = None

        for prov_name in order:
            if prov_name not in self.providers:
                continue

            try:
                logger.info(f"尝试使用 {prov_name} 生成")
                response = await self.providers[prov_name].generate(prompt)
                logger.info(f"{prov_name} 生成成功")
                return response, prov_name
            except Exception as e:
                logger.warning(f"{prov_name} 生成失败: {e}")
                last_error = e
                continue

        raise Exception(f"所有LLM提供者都失败: {last_error}")

    def _format_market_data(self, stock_data: StockData) -> str:
        """格式化市场数据"""
        lines = []

        # 实时行情
        if stock_data.realtime:
            rt = stock_data.realtime
            lines.append(f"- 最新价: {rt.price}")
            lines.append(f"- 涨跌幅: {rt.change_pct:.2f}%")
            lines.append(f"- 今开: {rt.open}")
            lines.append(f"- 最高: {rt.high}")
            lines.append(f"- 最低: {rt.low}")
            lines.append(f"- 成交量: {rt.volume}")
            lines.append(f"- 换手率: {rt.turnover_rate:.2f}%")

        # 历史数据摘要
        if stock_data.historical is not None and not stock_data.historical.empty:
            df = stock_data.historical
            lines.append(f"- 近{len(df)}日数据")

            # 计算简单指标
            if "Close" in df.columns:
                latest_close = df["Close"].iloc[-1]
                ma5 = df["Close"].rolling(5).mean().iloc[-1]
                ma20 = df["Close"].rolling(20).mean().iloc[-1]
                lines.append(f"- 5日均线: {ma5:.2f}")
                lines.append(f"- 20日均线: {ma20:.2f}")

        return "\n".join(lines)

    def _format_technical_indicators(self, stock_data: StockData) -> str:
        """格式化技术指标"""
        lines = []

        if stock_data.historical is not None and not stock_data.historical.empty:
            df = stock_data.historical

            # MA
            if "Close" in df.columns:
                close = df["Close"]
                lines.append(f"- MA5: {close.rolling(5).mean().iloc[-1]:.2f}")
                lines.append(f"- MA10: {close.rolling(10).mean().iloc[-1]:.2f}")
                lines.append(f"- MA20: {close.rolling(20).mean().iloc[-1]:.2f}")
                lines.append(f"- MA60: {close.rolling(60).mean().iloc[-1]:.2f}" if len(df) >= 60 else "")

        return "\n".join(lines)

    def _format_chip_data(self, stock_data: StockData) -> str:
        """格式化筹码数据"""
        lines = []

        if stock_data.chip_distribution:
            chip = stock_data.chip_distribution
            lines.append(f"- 获利盘比例: {chip.profit_ratio:.2f}%")
            lines.append(f"- 平均成本: {chip.avg_cost:.2f}")
            lines.append(f"- 90%筹码集中度: {chip.concentration_ratio_90:.2f}")

        return "\n".join(lines) if lines else "暂无筹码数据"

    def _format_search_info(self, search_info: SearchInfo) -> str:
        """格式化搜索信息"""
        lines = []

        if search_info.news:
            lines.append("### 相关新闻")
            for news in search_info.news[:3]:
                lines.append(f"- {news.title}")
                if news.snippet:
                    lines.append(f"  {news.snippet[:100]}...")

        if search_info.announcements:
            lines.append("\n### 公司公告")
            for ann in search_info.announcements[:3]:
                lines.append(f"- {ann.title}")

        return "\n".join(lines) if lines else "暂无相关资讯"

    def _parse_analysis_response(self, response: str) -> Dict[str, Any]:
        """解析分析响应"""
        result = {
            "fundamental_analysis": "",
            "technical_analysis": "",
            "chip_analysis": "",
            "news_analysis": "",
            "rating": "中性",
            "target_price": None,
            "stop_loss": None,
            "take_profit": None,
            "risks": [],
            "operation_advice": ""
        }

        # 简单解析：提取各个部分
        sections = {
            "基本面分析": "fundamental_analysis",
            "技术面分析": "technical_analysis",
            "筹码分析": "chip_analysis",
            "消息面分析": "news_analysis",
            "投资建议": "operation_advice",
            "操作建议": "operation_advice"
        }

        current_section = None
        current_content = []

        for line in response.split("\n"):
            # 检测章节标题
            for section_name, field_name in sections.items():
                if section_name in line:
                    if current_section and current_content:
                        result[current_section] = "\n".join(current_content).strip()
                    current_section = field_name
                    current_content = []
                    break
            else:
                if current_section:
                    current_content.append(line)

            # 检测评级
            if "强烈推荐" in line:
                result["rating"] = "强烈推荐"
            elif "推荐" in line and "不推荐" not in line:
                result["rating"] = "推荐"
            elif "不推荐" in line and "强烈不推荐" not in line:
                result["rating"] = "不推荐"
            elif "强烈不推荐" in line:
                result["rating"] = "强烈不推荐"

        # 保存最后的内容
        if current_section and current_content:
            result[current_section] = "\n".join(current_content).strip()

        return result

    async def analyze_stock(
        self,
        stock_data: StockData,
        search_info: SearchInfo = None,
        stock_name: str = ""
    ) -> AnalysisReport:
        """
        分析股票

        Args:
            stock_data: 股票数据
            search_info: 搜索信息
            stock_name: 股票名称

        Returns:
            AnalysisReport: 分析报告
        """
        logger.info(f"开始分析股票: {stock_data.symbol}")

        # 构建提示
        prompt = ANALYSIS_PROMPT_TEMPLATE.format(
            symbol=stock_data.symbol,
            stock_name=stock_name or stock_data.symbol,
            market_data=self._format_market_data(stock_data),
            technical_indicators=self._format_technical_indicators(stock_data),
            chip_data=self._format_chip_data(stock_data),
            search_info=self._format_search_info(search_info) if search_info else "暂无相关资讯"
        )

        # 调用LLM
        response, provider = await self.generate(prompt)

        # 解析响应
        parsed = self._parse_analysis_response(response)

        # 创建报告
        report = AnalysisReport(
            symbol=stock_data.symbol,
            stock_name=stock_name,
            fundamental_analysis=parsed.get("fundamental_analysis", ""),
            technical_analysis=parsed.get("technical_analysis", ""),
            chip_analysis=parsed.get("chip_analysis", ""),
            news_analysis=parsed.get("news_analysis", ""),
            rating=parsed.get("rating", "中性"),
            target_price=parsed.get("target_price"),
            stop_loss=parsed.get("stop_loss"),
            take_profit=parsed.get("take_profit"),
            risks=parsed.get("risks", []),
            operation_advice=parsed.get("operation_advice", ""),
            raw_response=response,
            model_used=provider
        )

        logger.info(f"股票 {stock_data.symbol} 分析完成，评级: {report.rating}")

        return report

    def analyze_stock_sync(
        self,
        stock_data: StockData,
        search_info: SearchInfo = None,
        stock_name: str = ""
    ) -> AnalysisReport:
        """同步分析股票"""
        return asyncio.run(self.analyze_stock(stock_data, search_info, stock_name))
