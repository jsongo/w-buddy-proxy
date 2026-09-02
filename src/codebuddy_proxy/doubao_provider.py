"""豆包（Doubao）provider —— 自研内联实现。

通过内联的 ``doubao.cdp_client.CDPDoubaoClient``（纯 stdlib CDP 直连豆包工作
App 的内置 Chromium，在页面 JS 环境里 fetch 自动注入 a_bogus 风控签名）直连
豆包，不再依赖外部 doubao2api HTTP 服务，也不依赖 Playwright。

本 provider 负责：
1. 持有并管理 CDPDoubaoClient 生命周期（登录态、健康检查）
2. 把 OpenAI 标准 chat 请求转成豆包文本提示词 + use_deep_think 模式
3. 把豆包原始 SSE 事件转回 OpenAI 标准格式（流式 / 非流式）

注意：豆包客户端模型不支持 function calling / tool use，仅适合对话、写作
场景，不能作为 coding agent 后端。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, AsyncIterator, Sequence

from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from .providers import BaseProvider

log = logging.getLogger(__name__)

# 豆包文本对话模型 -> use_deep_think 模式（0=快速, 1=思考, 3=专家）
_DOUBAO_CHAT_MODELS: dict[str, tuple[str, int]] = {
    "doubao": ("豆包快速模式", 0),
    "doubao-pro": ("豆包 Pro", 0),
    "doubao-think": ("豆包深度思考", 1),
    "doubao-expert": ("豆包专家模式", 3),
}


def _extract_prompt(messages: list[dict[str, Any]]) -> str:
    """从 OpenAI 标准 messages 提取文本提示词。

    豆包没有原生多轮 messages API，多轮时把历史拼成 ``[role]: content``
    文本（与 doubao2api 原实现一致）。
    """
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "".join(
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        else:
            text = ""
        if text:
            # user 消息保持自然语言（不带前缀）；非 user（system 等）必须带
            # role 前缀，否则单条 system 消息会被当成用户正文发给豆包
            if role == "user":
                parts.append(text)
            else:
                parts.append(f"[{role}]: {text}")
    return "\n".join(parts)


class DoubaoProvider(BaseProvider):
    id = "doubao"
    name = "豆包 (自研内联)"

    def __init__(
        self,
        startup_timeout: float = 120.0,
    ):
        from .doubao.cdp_client import CDPDoubaoClient

        self._client = CDPDoubaoClient()
        self._startup_timeout = startup_timeout
        self._started = False
        # 防止并发首请求双重启动（双开 Helper / 重复连接 CDP）
        self._start_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # BaseProvider 接口
    # ------------------------------------------------------------------

    def models(self) -> Sequence[dict[str, Any]]:
        return [
            {
                "id": mid,
                "object": "model",
                "owned_by": self.id,
                "created": 0,
                "description": desc,
            }
            for mid, (desc, _) in _DOUBAO_CHAT_MODELS.items()
        ]

    def ensure_auth(self) -> None:
        """豆包认证是异步的（CDP 浏览器登录态），无法在同步方法里可靠
        检查。这里只做软检查（no-op），真正的启动 + 登录态检查由 ``forward``
        里的 ``_ensure_started`` 完成，未登录时由 ``forward`` 抛 401。
        """
        return

    async def _ensure_started(self) -> None:
        """异步确保 CDPDoubaoClient 已启动（惰性启动，加锁防并发双重启动）。"""
        if self._started:
            return
        async with self._start_lock:
            if self._started:  # 双重检查：等锁期间可能已被并发请求启动
                return
            log.info("DoubaoProvider: starting CDP client")
            try:
                await self._client.start()
                self._started = True
            except Exception as exc:
                log.error("DoubaoProvider: client start failed: %s", exc)
                raise HTTPException(
                    status_code=502,
                    detail=f"doubao client start failed: {exc}",
                ) from exc

    async def forward(
        self,
        body: dict[str, Any],
        protocol: str,
        original: dict[str, Any] | None = None,
    ) -> StreamingResponse | JSONResponse:
        await self._ensure_started()

        if not self._client.is_ready:
            raise HTTPException(
                status_code=401,
                detail="doubao not logged in - 请先完成豆包扫码登录",
            )

        requested_model = body.get("model", "doubao")
        if requested_model not in _DOUBAO_CHAT_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"unknown doubao model '{requested_model}'",
            )
        _, use_deep_think = _DOUBAO_CHAT_MODELS[requested_model]

        messages = body.get("messages", [])
        prompt = _extract_prompt(messages)
        if not prompt:
            raise HTTPException(status_code=400, detail="no text content")

        conversation_id = body.get("conversation_id")
        bot_id = body.get("bot_id")

        if body.get("stream"):
            return StreamingResponse(
                self._stream(prompt, use_deep_think, requested_model, conversation_id, bot_id),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "close"},
            )

        message = await self._collect(prompt, use_deep_think, conversation_id, bot_id)
        return JSONResponse(content=self._build_nonstream_response(message, requested_model))

    def health(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "started": self._started,
            "ready": self._client.is_ready,
        }

    # ------------------------------------------------------------------
    # 转换逻辑
    # ------------------------------------------------------------------

    def _build_nonstream_response(
        self, message: dict[str, Any], model: str
    ) -> dict[str, Any]:
        content = message.get("content", "")
        reasoning = message.get("reasoning_content", "")
        resp: dict[str, Any] = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
        if reasoning:
            resp["choices"][0]["message"]["reasoning_content"] = reasoning
        if message.get("conversation_id"):
            resp["conversation_id"] = message["conversation_id"]
        return resp

    async def _stream(
        self,
        prompt: str,
        use_deep_think: int,
        model: str,
        conversation_id: str | None,
        bot_id: str | None,
    ) -> AsyncIterator[str]:
        """把豆包原始 SSE 事件转成 OpenAI chat.completion.chunk 流。"""
        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        thinking_count = 0
        in_thinking = False
        result_conversation_id = conversation_id

        def _chunk(delta: dict[str, Any], finish_reason: str | None = None) -> dict[str, Any]:
            return {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }

        def _iter_blocks(data: dict[str, Any]):
            for patch in data.get("patch_op", []):
                pv = patch.get("patch_value", {})
                yield from pv.get("content_block", [])
            dc = data.get("content", {})
            if isinstance(dc, dict):
                yield from dc.get("content_block", [])

        try:
            async for event in self._client.chat_completion(
                prompt,
                use_deep_think=use_deep_think,
                conversation_id=conversation_id or None,
                bot_id=bot_id or None,
            ):
                if event.get("error"):
                    status = event.get("status", 0)
                    body_text = event.get("body", "")
                    log.error("doubao stream error %s: %s", status, body_text[:200])
                    self._client.record_failure(status or 0)
                    yield f"data: {json.dumps(_chunk({'content': f'[Error {status}]'}), ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                event_type = event.get("_event", "")

                if not result_conversation_id:
                    cid = self._client.extract_conversation_id(event)
                    if cid and cid != "0":
                        result_conversation_id = cid

                # 风控 / 会话过期错误
                if event_type == "STREAM_ERROR" or event.get("error_code"):
                    code = event.get("error_code", 0)
                    msg = event.get("error_msg", "unknown error")
                    self._client.record_failure(code)
                    yield f"data: {json.dumps(_chunk({'content': f'[Error code={code}: {msg}]'}), ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                # CHUNK_DELTA：紧凑 {"text": "..."}
                if (
                    event_type == "CHUNK_DELTA"
                    and isinstance(event.get("text"), str)
                    and event["text"]
                ):
                    t = event["text"]
                    if in_thinking:
                        yield f"data: {json.dumps(_chunk({'reasoning_content': t}), ensure_ascii=False)}\n\n"
                    else:
                        yield f"data: {json.dumps(_chunk({'role': 'assistant', 'content': t}), ensure_ascii=False)}\n\n"
                    continue

                # content_block 数组（thinking 标记 / 正文）
                for cb in _iter_blocks(event):
                    bt = cb.get("block_type", 0)
                    block_content = cb.get("content", {})

                    if bt == 10040:
                        thinking_count += 1
                        in_thinking = thinking_count == 1
                        continue

                    if bt == 10000:
                        tb = block_content.get("text_block", {})
                        if isinstance(tb, dict) and tb.get("text"):
                            t = tb["text"]
                            if in_thinking:
                                yield f"data: {json.dumps(_chunk({'reasoning_content': t}), ensure_ascii=False)}\n\n"
                            else:
                                yield f"data: {json.dumps(_chunk({'role': 'assistant', 'content': t}), ensure_ascii=False)}\n\n"

        except Exception as exc:
            log.error("doubao stream error: %s", exc)
            self._client.record_failure(0)
            yield f"data: {json.dumps(_chunk({'content': f'[Error: {exc}]'}), ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            return

        self._client.record_success()

        final_delta: dict[str, Any] = {}
        if result_conversation_id:
            final_delta["conversation_id"] = result_conversation_id
        yield f"data: {json.dumps(_chunk(final_delta, 'stop'), ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    async def _collect(
        self,
        prompt: str,
        use_deep_think: int,
        conversation_id: str | None,
        bot_id: str | None,
    ) -> dict[str, Any]:
        """非流式：聚合完整响应，分离 reasoning_content 与 content。"""
        thinking_count = 0
        in_thinking = False
        thinking_parts: list[str] = []
        content_parts: list[str] = []
        result_conversation_id = conversation_id

        def _iter_blocks(data: dict[str, Any]):
            for patch in data.get("patch_op", []):
                pv = patch.get("patch_value", {})
                yield from pv.get("content_block", [])
            dc = data.get("content", {})
            if isinstance(dc, dict):
                yield from dc.get("content_block", [])

        try:
            async for event in self._client.chat_completion(
                prompt,
                use_deep_think=use_deep_think,
                conversation_id=conversation_id or None,
                bot_id=bot_id or None,
            ):
                if event.get("error"):
                    self._client.record_failure(event.get("status") or 0)
                    raise RuntimeError(
                        f"API error {event.get('status')}: {event.get('body', '')[:200]}"
                    )
                if event.get("error_code"):
                    code = event.get("error_code", 0)
                    msg = event.get("error_msg", "")
                    self._client.record_failure(code)
                    raise RuntimeError(f"Error code={code}: {msg}")

                if not result_conversation_id:
                    cid = self._client.extract_conversation_id(event)
                    if cid and cid != "0":
                        result_conversation_id = cid

                event_type = event.get("_event", "")

                if (
                    event_type == "CHUNK_DELTA"
                    and isinstance(event.get("text"), str)
                    and event["text"]
                ):
                    if in_thinking:
                        thinking_parts.append(event["text"])
                    else:
                        content_parts.append(event["text"])
                    continue

                for cb in _iter_blocks(event):
                    bt = cb.get("block_type", 0)
                    block_content = cb.get("content", {})

                    if bt == 10040:
                        thinking_count += 1
                        in_thinking = thinking_count == 1
                    elif bt == 10000:
                        tb = block_content.get("text_block", {})
                        if isinstance(tb, dict) and tb.get("text"):
                            if in_thinking:
                                thinking_parts.append(tb["text"])
                            else:
                                content_parts.append(tb["text"])

        except RuntimeError:
            raise

        self._client.record_success()

        message: dict[str, Any] = {"role": "assistant", "content": "".join(content_parts)}
        if thinking_parts:
            message["reasoning_content"] = "".join(thinking_parts)
        if result_conversation_id:
            message["conversation_id"] = result_conversation_id
        return message

    # ------------------------------------------------------------------
    # 登录辅助（供外部触发扫码）
    # ------------------------------------------------------------------

    async def wait_for_login(self, timeout: float = 120.0) -> bool:
        """等待用户扫码登录（供管理端/CLI 调用）。"""
        await self._ensure_started()
        return await self._client.wait_for_login(timeout=timeout)

    @property
    def page_url(self) -> str:
        page = getattr(self._client, "page", None)
        return page.url if page else ""
