"""Trae SSE 解析与 Anthropic 流包装。"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Iterator

from ..anthropic_adapter import AnthropicStreamConverter

log = logging.getLogger(__name__)

# ───────────────────────── SSE 解析 ─────────────────────────

# Trae 错误码 -> 友好中文文案（官方 docs.trae.ai/ide/error-codes）
_TRAE_ERROR_HINTS: dict[int, str] = {
    1005: "Trae plan 权益不足（当前账号/模型组合无权限，可检查会员套餐或换模型）",
    3003: "Trae 模型暂不可用（今日额度可能已用完，或当前模型无权限）",
    3004: "Trae 当前模型访问量过大，请稍后重试",
    4001: "Trae 服务端错误，请稍后重试",
    4007: "Trae 请求限流，请稍后重试",
    4010: "Trae 检测到风险账号，已自动登出，请重新登录",
    4011: "Trae AI 问答今日用量已达上限，请明日再试",
    4013: "Trae AI 服务在当前地区不可用",
    4015: "Trae 检测到账号/IP 风险，请求已被阻止",
    4021: "Trae 今日会话次数已达上限，请明日再试",
    4022: "Trae 请求失败，请尝试新建会话并重试",
    4023: "Trae 模型列表已更新，请确认后重试",
    4031: "Trae 今日请求额度已用完，请明日再试（每日 0 点重置）",
    4050: "Trae 请求超时，模型服务资源紧张，请稍后重试",
    4051: "Trae 请求超时，模型服务资源紧张，请稍后重试",
}


def _trae_error_text(data: dict[str, Any]) -> str:
    """把 Trae 错误事件转成友好中文文案（附错误码，便于定位）。"""
    code = data.get("code")
    msg = data.get("message") or data.get("error") or ""
    if isinstance(code, int) and code in _TRAE_ERROR_HINTS:
        hint = _TRAE_ERROR_HINTS[code]
        text = hint + (f"（{msg}）" if msg and msg not in hint else "")
    elif msg:
        text = f"Trae 错误: {msg}"
    else:
        text = "Trae 未知错误"
    if code is not None and str(code) not in text:
        text = f"{text} (code: {code})"
    return text

def _parse_sse(text: str) -> list[tuple[str, dict[str, Any]]]:
    """解析 Trae SSE 流 -> [(event, data_dict), ...]。"""
    events = []
    current_event = ""
    current_data: list[str] = []

    def flush():
        if current_data:
            data_str = "\n".join(current_data)
            try:
                parsed = json.loads(data_str)
            except Exception:
                parsed = {"raw": data_str}
            events.append((current_event, parsed))
            current_data.clear()

    for line in text.splitlines():
        line = line.strip()
        if not line:
            flush()
            current_event = ""
        elif line.startswith("event:"):
            current_event = line[6:].strip()
        elif line.startswith("data:"):
            current_data.append(line[5:].strip())
    flush()
    return events


# ───────────────────────── Provider 实现 ─────────────────────────

def _anthropic_sse(event_name: str, payload: dict[str, Any]) -> str:
    """构造一条 Anthropic SSE 事件（event + data）。"""
    return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _wrap_anthropic_stream(
    openai_stream: Iterator[str], model: str
) -> Iterator[str]:
    """把 TraeProvider._stream 的 OpenAI chat chunk SSE 流包成 Anthropic 事件流。

    /v1/messages（Claude Code 等 Anthropic 客户端）经 anthropic_to_chat 转成
    chat 请求后路由到 Trae；本包装器把 OpenAI chunk 喂给
    AnthropicStreamConverter，输出 message_start / content_block_delta /
    message_stop 等标准事件（含 reasoning_content → thinking 块、
    tool_calls → tool_use 块）。_stream 是同步生成器，此处保持同步迭代。
    """
    converter = AnthropicStreamConverter(model)
    for piece in openai_stream:
        for line in piece.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                continue
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(chunk, dict) and chunk.get("error"):
                # 错误 chunk → Anthropic error 事件（而非塞进正文文本）
                err = chunk["error"]
                msg = str(err.get("message", err))
                code = err.get("code")
                if code is not None and str(code) not in msg:
                    msg = f"{msg} (code: {code})"
                yield _anthropic_sse("error", {
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": msg,
                    },
                })
                return
            for event_name, payload in converter.feed_chunk(chunk):
                yield _anthropic_sse(event_name, payload)
    for event_name, payload in converter.finish():
        yield _anthropic_sse(event_name, payload)
    # 与 CodeBuddy 通道的 anthropic 流保持一致：message_stop 后补 [DONE]
    # 确保 SSE 客户端立即结束等待
    yield "data: [DONE]\n\n"

