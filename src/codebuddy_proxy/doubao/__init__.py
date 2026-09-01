"""豆包（Doubao）聊天子包 —— 自研内联实现。

提供 ``CDPDoubaoClient``（纯 stdlib CDP，直连豆包工作 App 的内置 Chromium，
零第三方依赖）。Playwright 网页版方案（``browser_client``）已移除。
"""

from .cdp_client import CDPDoubaoClient

DEFAULT_BOT_ID = "7338286299411103781"

__all__ = ["CDPDoubaoClient", "DEFAULT_BOT_ID"]
