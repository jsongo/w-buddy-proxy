"""豆包（Doubao）provider。

通过本地 doubao2api 服务（wangchuxiaoji-oss/doubao2api）接入豆包模型。
doubao2api 已经暴露标准 OpenAI 兼容接口（/v1/chat/completions），因此本
provider 是「标准 OpenAI → 标准 OpenAI」的薄透传，只需要：

1. 把请求转发到 doubao2api 的 base_url
2. 探测 doubao2api 健康状态（登录态、风控）

注意：豆包客户端模型不支持 function calling / tool use，仅适合对话、写作、
多模态生成场景，不能作为 coding agent 后端。
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import httpx
from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from .providers import BaseProvider

log = logging.getLogger(__name__)

# doubao2api 暴露的文本对话模型（走 /v1/chat/completions）。
# 注意：doubao-image / doubao-video / doubao-music 走专用端点
# （/v1/images/generations 等），不属于 chat 补全，故不在此列出。
_DOUBAO_CHAT_MODELS = [
    ("doubao", "豆包快速模式"),
    ("doubao-pro", "豆包 Pro"),
    ("doubao-think", "豆包深度思考"),
    ("doubao-expert", "豆包专家模式"),
]


class DoubaoProvider(BaseProvider):
    id = "doubao"
    name = "豆包 (doubao2api)"

    def __init__(self, base_url: str, timeout: float = 300.0):
        # base_url 形如 http://127.0.0.1:9090/v1
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._http = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=10.0),
            # doubao2api 本地服务无需鉴权 key
        )

    def models(self) -> Sequence[dict[str, Any]]:
        return [
            {
                "id": mid,
                "object": "model",
                "owned_by": self.id,
                "created": 0,
                "description": desc,
            }
            for mid, desc in _DOUBAO_CHAT_MODELS
        ]

    async def _probe_health(self) -> dict[str, Any]:
        """探测 doubao2api 的 /health，返回其 JSON（失败返回空 dict）。"""
        try:
            # health 端点不带 /v1 前缀
            health_url = self.base_url.rsplit("/v1", 1)[0] + "/health"
            resp = await self._http.get(health_url)
            if resp.status_code == 200:
                return resp.json()
        except Exception as exc:
            log.warning("doubao2api health probe failed: %s", exc)
        return {}

    def ensure_auth(self) -> None:
        """豆包认证由 doubao2api 内部管理（浏览器登录态）。

        此处仅做同步的软检查：无法可靠同步探测登录态，故不阻塞；
        真正的认证失败会体现在 doubao2api 返回的非 200 状态上，
        由 forward 透传错误。若 doubao2api 完全不可达，forward 会
        返回 502。
        """
        # 无同步可做的检查；登录态在首次转发时由 doubao2api 返回错误暴露。
        return

    async def forward(
        self,
        body: dict[str, Any],
        protocol: str,
        original: dict[str, Any] | None = None,
    ) -> StreamingResponse | JSONResponse:
        # 豆包上游即标准 OpenAI 格式，无需额外转换；仅透传。
        # 但 responses / anthropic 协议客户端需要的是对应格式输出，
        # 这里统一按 openai 透传（调用方 __main__ 负责协议转换）。
        stream = bool(body.get("stream"))
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}

        if stream:
            return StreamingResponse(
                self._stream(url, headers, body),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "close"},
            )
        else:
            collected = await self._collect(url, headers, body)
            return JSONResponse(content=collected)

    async def _stream(self, url: str, headers: dict[str, str], body: dict[str, Any]):
        """流式透传 doubao2api 的 SSE 响应。"""
        async with self._http.stream("POST", url, headers=headers, json=body) as resp:
            if resp.status_code != 200:
                error_text = (await resp.aread()).decode("utf-8", "replace")
                # 尝试按 SSE error 格式吐出，客户端可解析
                yield f'data: {{"error": {{"message": "doubao upstream {resp.status_code}", "detail": "{error_text[:200]}"}}}}\n\n'
                yield "data: [DONE]\n\n"
                return
            async for line in resp.aiter_lines():
                if line:
                    yield line + "\n"

    async def _collect(self, url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
        """非流式：转发并返回完整 JSON。"""
        async with self._http.stream("POST", url, headers=headers, json=body) as resp:
            raw = await resp.aread()
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=502,
                    detail={
                        "error": {
                            "message": f"doubao upstream returned {resp.status_code}",
                            "type": "upstream_error",
                            "details": raw.decode("utf-8", "replace")[:300],
                        }
                    },
                )
            import json

            return json.loads(raw.decode("utf-8", "replace"))
