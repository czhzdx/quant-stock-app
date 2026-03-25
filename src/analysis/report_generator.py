"""报告生成器模块"""
import os
import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.analysis.llm_analyzer import AnalysisReport
from src.analysis.data_enhanced import StockData
from src.utils.config_loader import get_config

logger = logging.getLogger(__name__)


class ReportGenerator:
    """报告生成器"""

    def __init__(self, output_dir: str = None):
        """
        初始化报告生成器

        Args:
            output_dir: 输出目录
        """
        self.config = get_config()
        self.report_config = self.config.get("stock_analysis", {}).get("reports", {})

        self.output_dir = Path(output_dir or self.report_config.get("output_dir", "output/reports"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.formats = self.report_config.get("formats", ["markdown", "json"])

    def generate_single_report(
        self,
        symbol: str,
        stock_data: StockData,
        analysis: AnalysisReport,
        stock_name: str = ""
    ) -> Dict[str, Path]:
        """
        生成单股报告

        Args:
            symbol: 股票代码
            stock_data: 股票数据
            analysis: 分析报告
            stock_name: 股票名称

        Returns:
            生成的文件路径字典
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{symbol}_{timestamp}"
        files = {}

        # Markdown报告
        if "markdown" in self.formats:
            md_content = self._generate_markdown(symbol, stock_data, analysis, stock_name)
            md_path = self.output_dir / f"{base_name}.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            files["markdown"] = md_path
            logger.info(f"生成Markdown报告: {md_path}")

        # JSON报告
        if "json" in self.formats:
            json_content = self._generate_json(symbol, stock_data, analysis, stock_name)
            json_path = self.output_dir / f"{base_name}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_content, f, ensure_ascii=False, indent=2)
            files["json"] = json_path
            logger.info(f"生成JSON报告: {json_path}")

        return files

    def generate_summary_report(
        self,
        analyses: List[AnalysisReport],
        output_name: str = None
    ) -> Dict[str, Path]:
        """
        生成汇总报告

        Args:
            analyses: 分析报告列表
            output_name: 输出文件名

        Returns:
            生成的文件路径字典
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = output_name or f"summary_{timestamp}"
        files = {}

        # Markdown汇总
        if "markdown" in self.formats:
            md_content = self._generate_summary_markdown(analyses)
            md_path = self.output_dir / f"{base_name}.md"
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)
            files["markdown"] = md_path
            logger.info(f"生成汇总Markdown报告: {md_path}")

        # JSON汇总
        if "json" in self.formats:
            json_content = self._generate_summary_json(analyses)
            json_path = self.output_dir / f"{base_name}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_content, f, ensure_ascii=False, indent=2)
            files["json"] = json_path
            logger.info(f"生成汇总JSON报告: {json_path}")

        return files

    def _generate_markdown(
        self,
        symbol: str,
        stock_data: StockData,
        analysis: AnalysisReport,
        stock_name: str
    ) -> str:
        """生成Markdown格式报告"""
        lines = []

        # 标题
        lines.append(f"# {stock_name or symbol} 股票分析报告")
        lines.append(f"\n> 生成时间: {analysis.analysis_time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"> 分析模型: {analysis.model_used}")
        lines.append("")

        # 基本信息
        lines.append("## 基本信息")
        lines.append(f"- **股票代码**: {symbol}")
        lines.append(f"- **股票名称**: {stock_name or '未知'}")

        if stock_data.realtime:
            rt = stock_data.realtime
            lines.append(f"- **最新价**: {rt.price}")
            lines.append(f"- **涨跌幅**: {rt.change_pct:.2f}%")
            lines.append(f"- **成交量**: {rt.volume:,}")
            lines.append(f"- **换手率**: {rt.turnover_rate:.2f}%")

        lines.append("")

        # 评级
        lines.append("## 投资评级")
        rating_emoji = {
            "强烈推荐": "🔥",
            "推荐": "👍",
            "中性": "😐",
            "不推荐": "👎",
            "强烈不推荐": "❌"
        }
        emoji = rating_emoji.get(analysis.rating, "")
        lines.append(f"### {emoji} {analysis.rating}")
        lines.append("")

        # 目标价位
        if analysis.target_price or analysis.stop_loss or analysis.take_profit:
            lines.append("### 价格目标")
            if analysis.target_price:
                lines.append(f"- **目标价**: {analysis.target_price}")
            if analysis.stop_loss:
                lines.append(f"- **止损位**: {analysis.stop_loss}")
            if analysis.take_profit:
                lines.append(f"- **止盈位**: {analysis.take_profit}")
            lines.append("")

        # 基本面分析
        if analysis.fundamental_analysis:
            lines.append("## 基本面分析")
            lines.append(analysis.fundamental_analysis)
            lines.append("")

        # 技术面分析
        if analysis.technical_analysis:
            lines.append("## 技术面分析")
            lines.append(analysis.technical_analysis)
            lines.append("")

        # 筹码分析
        if analysis.chip_analysis:
            lines.append("## 筹码分析")
            lines.append(analysis.chip_analysis)
            lines.append("")

        # 消息面分析
        if analysis.news_analysis:
            lines.append("## 消息面分析")
            lines.append(analysis.news_analysis)
            lines.append("")

        # 操作建议
        if analysis.operation_advice:
            lines.append("## 操作建议")
            lines.append(analysis.operation_advice)
            lines.append("")

        # 风险提示
        if analysis.risks:
            lines.append("## 风险提示")
            for risk in analysis.risks:
                lines.append(f"- {risk}")
            lines.append("")

        # 免责声明
        lines.append("---")
        lines.append("")
        lines.append("### 免责声明")
        lines.append("本报告由AI自动生成，仅供参考，不构成投资建议。投资有风险，入市需谨慎。")

        return "\n".join(lines)

    def _generate_json(
        self,
        symbol: str,
        stock_data: StockData,
        analysis: AnalysisReport,
        stock_name: str
    ) -> dict:
        """生成JSON格式报告"""
        return {
            "symbol": symbol,
            "stock_name": stock_name,
            "analysis_time": analysis.analysis_time.isoformat(),
            "model_used": analysis.model_used,
            "rating": analysis.rating,
            "target_price": analysis.target_price,
            "stop_loss": analysis.stop_loss,
            "take_profit": analysis.take_profit,
            "fundamental_analysis": analysis.fundamental_analysis,
            "technical_analysis": analysis.technical_analysis,
            "chip_analysis": analysis.chip_analysis,
            "news_analysis": analysis.news_analysis,
            "operation_advice": analysis.operation_advice,
            "risks": analysis.risks,
            "market_data": stock_data.realtime.to_dict() if stock_data.realtime else None,
            "chip_data": stock_data.chip_distribution.to_dict() if stock_data.chip_distribution else None
        }

    def _generate_summary_markdown(self, analyses: List[AnalysisReport]) -> str:
        """生成汇总Markdown报告"""
        lines = []

        lines.append("# 股票分析汇总报告")
        lines.append(f"\n> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"> 分析股票数: {len(analyses)}")
        lines.append("")

        # 汇总表格
        lines.append("## 评级汇总")
        lines.append("")
        lines.append("| 股票代码 | 股票名称 | 评级 | 目标价 | 止损位 | 止盈位 |")
        lines.append("|----------|----------|------|--------|--------|--------|")

        rating_order = {"强烈推荐": 1, "推荐": 2, "中性": 3, "不推荐": 4, "强烈不推荐": 5}
        sorted_analyses = sorted(analyses, key=lambda x: rating_order.get(x.rating, 3))

        for analysis in sorted_analyses:
            lines.append(
                f"| {analysis.symbol} | {analysis.stock_name} | {analysis.rating} | "
                f"{analysis.target_price or '-'} | {analysis.stop_loss or '-'} | {analysis.take_profit or '-'} |"
            )

        lines.append("")

        # 各股详情
        lines.append("## 各股详情")
        lines.append("")

        for analysis in sorted_analyses:
            lines.append(f"### {analysis.stock_name or analysis.symbol}")
            lines.append(f"- **评级**: {analysis.rating}")
            if analysis.operation_advice:
                lines.append(f"- **操作建议**: {analysis.operation_advice[:200]}...")
            lines.append("")

        # 免责声明
        lines.append("---")
        lines.append("")
        lines.append("### 免责声明")
        lines.append("本报告由AI自动生成，仅供参考，不构成投资建议。投资有风险，入市需谨慎。")

        return "\n".join(lines)

    def _generate_summary_json(self, analyses: List[AnalysisReport]) -> dict:
        """生成汇总JSON报告"""
        rating_stats = {}
        for analysis in analyses:
            rating_stats[analysis.rating] = rating_stats.get(analysis.rating, 0) + 1

        return {
            "generation_time": datetime.now().isoformat(),
            "total_stocks": len(analyses),
            "rating_distribution": rating_stats,
            "analyses": [analysis.to_dict() for analysis in analyses]
        }
