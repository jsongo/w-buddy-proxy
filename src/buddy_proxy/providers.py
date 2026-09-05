"""Provider 抽象层。

buddy-proxy 现在支持多个上游源（CodeBuddy、豆包等），每个源是一个
Provider 实现。Provider 负责两件事：

1. 认证（各自的登录态管理，互不干扰）
2. 转发 chat 请求到上游并解析响应

接口设计原则：
- ``forward`` 接收标准 OpenAI chat 请求体（已做 tool_choice 归一化等预处理），
  返回 ``StreamingResponse``（流式）或 ``JSONResponse``（非流式），与现有
  ``__main__.py`` 的 ``forward_chat`` 语义对齐。
- 每个 provider 自持 HTTP 客户端与认证状态，不共享。
- 模型列表由 provider 各自声明，``/v1/models`` 合并输出。
"""

from __future__ import annotations

import abc
from typing import Any, Sequence

from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse


class BaseProvider(abc.ABC):
    """上游模型源的统一抽象。"""

    #: provider 唯一标识（用于日志、路由、模型归属）。
    #:
    #: 命名约定：只允许小写字母、数字和连字符 ``-``（形如 ``doubao``、
    #: ``doubao-pro``、``codebuddy``）。**禁止使用斜杠 ``/``**——因为 provider
    #: id 会作为 ``owned_by`` 暴露在 ``/v1/models`` 中，且可能被上层配置
    #: （如 ethan-ai 的 ``provider:`` 字段）直接引用，斜杠会与「provider/模型」
    #: 的层级分隔符产生歧义。
    id: str = "base"
    #: 人类可读名称。
    name: str = "Base Provider"

    @abc.abstractmethod
    def models(self) -> Sequence[dict[str, Any]]:
        """返回本 provider 提供的模型列表（OpenAI /v1/models 格式元素）。"""

    @abc.abstractmethod
    def ensure_auth(self) -> None:
        """确保已认证；失败时抛 ``HTTPException``（401）。"""

    @abc.abstractmethod
    async def forward(
        self,
        body: dict[str, Any],
        protocol: str,
        original: dict[str, Any] | None = None,
    ) -> StreamingResponse | JSONResponse:
        """转发 chat 请求到上游并返回响应。

        参数与 ``__main__.forward_chat`` 一致：
        - ``body``: 标准 OpenAI chat 请求体（已归一化、已设 stream）
        - ``protocol``: 客户端协议（"openai" / "responses" / "anthropic"），
          供 provider 决定是否做额外转换
        - ``original``: 原始请求体（协议转换前的，用于 responses/anthropic 还原）
        """

    def health(self) -> dict[str, Any]:
        """返回 provider 健康状态（合并进 /health）。"""
        return {"id": self.id, "name": self.name}

    # ------------------------------------------------------------------
    # 可选能力：打卡 / 额度（/ui 管理页消费；不支持时返回 None）
    # 全部为同步方法（上游是 urllib/httpx 同步调用），调用方用
    # asyncio.to_thread 包装，避免阻塞事件循环。
    # ------------------------------------------------------------------

    def checkin_status(self) -> dict[str, Any] | None:
        """查询今日签到状态。返回 ``{"checked_in": bool, "claimable": bool,
        "inactive": bool, "streak_days": int, "message": str}``（claimable/
        inactive 缺省视为 True/False）；不支持签到时返回 None。"""
        return None

    def checkin_claim(self) -> dict[str, Any] | None:
        """领取今日签到积分。返回格式同 checkin_status；
        失败时抛异常（由调用方转成错误信息）；不支持时返回 None。"""
        return None

    def quota(self) -> dict[str, Any] | None:
        """查询套餐额度。返回 ``{"items": [{label, used, total, remaining,
        percent, reset_ts}], "level": str|None}``（remaining/percent 无法
        计算时为 None）；不支持时返回 None。"""
        return None
