"""通知发送模块 - 邮件/微信"""
import os
import smtplib
import logging
import aiohttp
import asyncio
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Optional, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod

from src.utils.config_loader import get_config

logger = logging.getLogger(__name__)


@dataclass
class NotificationResult:
    """通知结果"""
    success: bool
    channel: str
    message: str
    error: str = None


class BaseNotifier(ABC):
    """通知器基类"""

    @abstractmethod
    async def send(self, title: str, content: str, **kwargs) -> NotificationResult:
        """发送通知"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """通知渠道名称"""
        pass


class EmailNotifier(BaseNotifier):
    """邮件通知器"""

    def __init__(
        self,
        smtp_server: str,
        smtp_port: int,
        sender: str,
        password: str,
        recipients: List[str]
    ):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender = sender
        self.password = password
        self.recipients = recipients

    @property
    def name(self) -> str:
        return "email"

    async def send(
        self,
        title: str,
        content: str,
        recipients: List[str] = None,
        html: bool = False
    ) -> NotificationResult:
        """发送邮件"""
        target_recipients = recipients or self.recipients

        if not target_recipients:
            return NotificationResult(
                success=False,
                channel="email",
                message="No recipients specified"
            )

        try:
            # 创建邮件
            msg = MIMEMultipart("alternative")
            msg["Subject"] = title
            msg["From"] = self.sender
            msg["To"] = ", ".join(target_recipients)

            # 添加内容
            if html:
                msg.attach(MIMEText(content, "html", "utf-8"))
            else:
                msg.attach(MIMEText(content, "plain", "utf-8"))

            # 发送
            def send_email():
                with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                    server.starttls()
                    server.login(self.sender, self.password)
                    server.sendmail(self.sender, target_recipients, msg.as_string())

            # 在线程池中执行
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, send_email)

            logger.info(f"邮件发送成功: {title}")
            return NotificationResult(
                success=True,
                channel="email",
                message=f"Email sent to {len(target_recipients)} recipients"
            )

        except Exception as e:
            logger.error(f"邮件发送失败: {e}")
            return NotificationResult(
                success=False,
                channel="email",
                message="Failed to send email",
                error=str(e)
            )


class WeChatNotifier(BaseNotifier):
    """微信通知器 (Server酱)"""

    def __init__(self, serverchan_key: str = None, webhook_url: str = None):
        self.serverchan_key = serverchan_key
        self.webhook_url = webhook_url

    @property
    def name(self) -> str:
        return "wechat"

    async def send(
        self,
        title: str,
        content: str
    ) -> NotificationResult:
        """发送微信通知"""
        # 优先使用Server酱
        if self.serverchan_key:
            return await self._send_via_serverchan(title, content)

        # 企业微信Webhook
        if self.webhook_url:
            return await self._send_via_webhook(title, content)

        return NotificationResult(
            success=False,
            channel="wechat",
            message="No WeChat configuration"
        )

    async def _send_via_serverchan(self, title: str, content: str) -> NotificationResult:
        """通过Server酱发送"""
        try:
            url = f"https://sctapi.ftqq.com/{self.serverchan_key}.send"

            async with aiohttp.ClientSession() as session:
                data = {
                    "title": title,
                    "desp": content
                }
                async with session.post(url, data=data) as response:
                    result = await response.json()

                    if result.get("code") == 0:
                        logger.info(f"Server酱发送成功: {title}")
                        return NotificationResult(
                            success=True,
                            channel="wechat",
                            message="Server酱发送成功"
                        )
                    else:
                        return NotificationResult(
                            success=False,
                            channel="wechat",
                            message=result.get("message", "Unknown error")
                        )

        except Exception as e:
            logger.error(f"Server酱发送失败: {e}")
            return NotificationResult(
                success=False,
                channel="wechat",
                message="Server酱发送失败",
                error=str(e)
            )

    async def _send_via_webhook(self, title: str, content: str) -> NotificationResult:
        """通过企业微信Webhook发送"""
        try:
            async with aiohttp.ClientSession() as session:
                data = {
                    "msgtype": "markdown",
                    "markdown": {
                        "content": f"## {title}\n\n{content}"
                    }
                }
                async with session.post(self.webhook_url, json=data) as response:
                    result = await response.json()

                    if result.get("errcode") == 0:
                        logger.info(f"企业微信发送成功: {title}")
                        return NotificationResult(
                            success=True,
                            channel="wechat",
                            message="企业微信发送成功"
                        )
                    else:
                        return NotificationResult(
                            success=False,
                            channel="wechat",
                            message=result.get("errmsg", "Unknown error")
                        )

        except Exception as e:
            logger.error(f"企业微信发送失败: {e}")
            return NotificationResult(
                success=False,
                channel="wechat",
                message="企业微信发送失败",
                error=str(e)
            )


class Notifier:
    """通知管理器"""

    def __init__(self):
        """初始化通知管理器"""
        self.config = get_config()
        self.notification_config = self.config.get("stock_analysis", {}).get("notifications", {})

        # 初始化通知器
        self.notifiers: Dict[str, BaseNotifier] = {}
        self._init_notifiers()

        # 启用的渠道
        self.enabled_channels = self.notification_config.get("channels", ["email", "wechat"])

    def _init_notifiers(self):
        """初始化通知器"""
        # 邮件通知器
        email_config = self.notification_config.get("email", {})
        if email_config.get("enabled"):
            smtp_server = self._get_config_value(email_config.get("smtp_server"))
            smtp_port = email_config.get("smtp_port", 587)
            sender = self._get_config_value(email_config.get("sender"))
            password = self._get_config_value(email_config.get("password"))
            recipients = email_config.get("recipients", [])

            if smtp_server and sender and password:
                self.notifiers["email"] = EmailNotifier(
                    smtp_server=smtp_server,
                    smtp_port=smtp_port,
                    sender=sender,
                    password=password,
                    recipients=recipients
                )

        # 微信通知器
        wechat_config = self.notification_config.get("wechat", {})
        if wechat_config.get("enabled"):
            serverchan_key = self._get_config_value(wechat_config.get("serverchan_key"))
            webhook_url = self._get_config_value(wechat_config.get("webhook_url"))

            if serverchan_key or webhook_url:
                self.notifiers["wechat"] = WeChatNotifier(
                    serverchan_key=serverchan_key,
                    webhook_url=webhook_url
                )

    def _get_config_value(self, value: str) -> Optional[str]:
        """获取配置值（支持环境变量）"""
        if not value:
            return None

        # 处理环境变量引用
        if value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            return os.environ.get(env_var)

        return value

    async def send(
        self,
        title: str,
        content: str,
        channels: List[str] = None,
        **kwargs
    ) -> List[NotificationResult]:
        """
        发送通知

        Args:
            title: 标题
            content: 内容
            channels: 指定渠道列表（None则使用全部启用的渠道）
            **kwargs: 其他参数

        Returns:
            通知结果列表
        """
        target_channels = channels or self.enabled_channels
        results = []

        for channel in target_channels:
            if channel in self.notifiers:
                try:
                    result = await self.notifiers[channel].send(title, content, **kwargs)
                    results.append(result)
                except Exception as e:
                    logger.error(f"通知发送异常 ({channel}): {e}")
                    results.append(NotificationResult(
                        success=False,
                        channel=channel,
                        message="Exception during send",
                        error=str(e)
                    ))
            else:
                logger.warning(f"未配置通知渠道: {channel}")

        return results

    def send_sync(
        self,
        title: str,
        content: str,
        channels: List[str] = None,
        **kwargs
    ) -> List[NotificationResult]:
        """同步发送通知"""
        return asyncio.run(self.send(title, content, channels, **kwargs))

    async def send_analysis_report(
        self,
        symbol: str,
        stock_name: str,
        rating: str,
        summary: str,
        channels: List[str] = None
    ) -> List[NotificationResult]:
        """
        发送分析报告通知

        Args:
            symbol: 股票代码
            stock_name: 股票名称
            rating: 评级
            summary: 摘要
            channels: 指定渠道

        Returns:
            通知结果列表
        """
        title = f"【{stock_name or symbol}】分析报告 - {rating}"
        content = f"""
## {stock_name or symbol} 分析报告

**评级**: {rating}

{summary}

---
*由量化选股系统自动生成*
"""
        return await self.send(title, content, channels)
