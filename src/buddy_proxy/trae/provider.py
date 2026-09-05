"""TraeProvider：OpenAI/Anthropic 兼容入口，编排原生通道与文本协议兜底。"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
import uuid
from typing import Any, AsyncIterator, Sequence

from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from ..anthropic_adapter import chat_completion_to_anthropic_message
from ..providers import BaseProvider
from .benefits_api import claim_checkin_credits, fetch_checkin_status, fetch_ent_usage
from .config import (
    MODEL_MAP,
    MODEL_TIERS,
    TRAE_HEARTBEAT_INTERVAL,
    _debug_dump,
    _heartbeat_text,
    _map_model,
)
from .credentials import _auth
from .leak_guard import _StreamLeakCleaner, _sanitize_agent_leak
from .native_tools import (
    _NativeToolAccumulator,
    _NATIVE_TOOLS_ENABLED,
    _native_rejected,
    _send_native_chat,
)
from .sse import _parse_sse, _trae_error_text, _wrap_anthropic_stream
from .text_protocol import _extract_prompt, _looks_like_agent_request
from .text_toolcall import _StreamToolCallSplitter, _parse_tool_calls, _tool_names
from .transport import send_trae_chat

log = logging.getLogger(__name__)

class TraeProvider(BaseProvider):
    id = "trae"
    name = "Trae (本地解密直连)"
    # 打卡/积分 API 只有 Trae 上游提供（/ui 自动打卡据此识别）
    supports_checkin = True

    def __init__(self, base_url: str | None = None, edition: str = "cn"):
        self._base_url = base_url or BASE_URL_CN
        self._edition = edition

    def models(self) -> Sequence[dict[str, Any]]:
        result = []
        seen = set()
        for tier, models in MODEL_TIERS.items():
            for m in models:
                if m in seen:
                    continue
                seen.add(m)
                result.append({
                    "id": m,
                    "object": "model",
                    "created": 0,
                    "owned_by": self.id,
                    "tier": tier,
                    "description": f"Trae {tier} 模型",
                })
        # 加别名（外部名映射）
        for external, internal in MODEL_MAP.items():
            if external not in seen:
                seen.add(external)
                result.append({
                    "id": external,
                    "object": "model",
                    "created": 0,
                    "owned_by": self.id,
                    "maps_to": internal,
                    "description": f"Trae 别名 -> {internal}",
                })
        return result

    def ensure_auth(self) -> None:
        _auth()

    # ---- 打卡 / 额度（/ui 管理页消费，均经 asyncio.to_thread 调用） ----

    def checkin_status(self) -> dict[str, Any] | None:
        data = fetch_checkin_status()
        checked_in = bool(data.get("checked_in"))
        return {
            "checked_in": checked_in,
            "claimable": bool(data.get("enable", True)) and not checked_in,
            "inactive": not bool(data.get("enable", True)),
            "message": data.get("message", ""),
        }

    def checkin_claim(self) -> dict[str, Any] | None:
        data = claim_checkin_credits()
        if data.get("code") not in (0, None):
            raise RuntimeError(data.get("message") or json.dumps(data, ensure_ascii=False)[:200])
        return {
            "checked_in": True,
            "extra_credits": data.get("credits_granted", data.get("extra_credits")),
            "message": data.get("message", ""),
        }

    def quota(self) -> dict[str, Any] | None:
        data = fetch_ent_usage()
        us = data.get("usage_summary", {})
        items: list[dict[str, Any]] = []
        total, consumed = us.get("total_amount"), us.get("consumed_amount")
        if total is not None:
            ratio = us.get("consumption_ratio")
            percent = round(ratio * 100) if isinstance(ratio, (int, float)) else None
            remaining = None
            if isinstance(total, (int, float)) and isinstance(consumed, (int, float)):
                remaining = round(total - consumed, 2)
            items.append({"label": "总额度", "used": consumed, "total": total,
                          "remaining": remaining, "percent": percent, "reset_ts": None})
        # 权益包列表可能带几十条历史"签到奖励"空记录，只保留有到期时间的前几个
        packs: list[dict[str, Any]] = []
        seen: set[str] = set()
        for p in data.get("user_entitlement_pack_list", []):
            eb = p.get("entitlement_base_info") or {}
            if not eb.get("end_time"):
                continue
            desc = p.get("display_desc") or "权益包"
            if desc in seen:
                continue
            seen.add(desc)
            packs.append({
                "label": desc,
                "used": None, "total": None, "percent": None,
                "reset_ts": eb.get("end_time"),
            })
            if len(packs) >= 3:
                break
        items.extend(packs)
        return {"items": items, "level": None}

    async def forward(
        self,
        body: dict[str, Any],
        protocol: str,
        original: dict[str, Any] | None = None,
    ) -> StreamingResponse | JSONResponse:
        requested_model = body.get("model", "auto")
        messages = body.get("messages", [])
        stream = bool(body.get("stream", False))
        # coding agent 请求：不注入 guard、不清洗——下游自己解析工具调用语法
        agent_mode = _looks_like_agent_request(messages, body)
        _debug_dump(
            "debug_trae_route",
            model=requested_model,
            agent_mode=agent_mode,
            tools_count=len(body.get("tools") or []),
            stream=stream,
            message_count=len(messages),
        )

        tools = body.get("tools") or []
        # 原生 function calling 通道：带 tools 的请求优先走结构化直通
        # （chat_v3 + 原生 tools），上游 4001 拒绝时自动回落文本协议。
        # prompt（文本协议转换）始终照算——它是兜底路径的输入。
        native_mode = bool(tools) and _NATIVE_TOOLS_ENABLED
        prompt = _extract_prompt(
            messages, guard=not agent_mode, tools=tools if agent_mode else None
        )
        if not prompt:
            raise HTTPException(status_code=400, detail="no text content")
        native = (
            {"messages": _native_messages(messages), "tools": tools}
            if native_mode else None
        )

        if stream:
            include_usage = bool(
                (body.get("stream_options") or {}).get("include_usage"))
            # anthropic 协议没有 stream_options 字段，但 message_delta 需要
            # usage（Claude Code 靠它统计 token），这里强制向 _stream 索取
            if protocol == "anthropic":
                include_usage = True
            gen = self._stream(prompt, requested_model,
                               sanitize=not agent_mode, tools=tools or None,
                               include_usage=include_usage, native=native)
            if protocol == "anthropic":
                gen = _wrap_anthropic_stream(gen, requested_model)
            return StreamingResponse(
                gen,
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "close"},
            )
        # 非流式聚合内部是同步 urllib 调用（最长 180s），放线程池执行，
        # 避免阻塞事件循环拖垮所有并发请求
        try:
            collected = await asyncio.to_thread(
                self._collect, prompt, requested_model, not agent_mode, tools or None,
                native,
            )
        except HTTPException as e:
            # anthropic 协议错误体用标准形状（Claude Code 等 SDK 靠它渲染错误）
            if protocol == "anthropic":
                return JSONResponse(
                    status_code=e.status_code,
                    content={
                        "type": "error",
                        "error": {"type": "api_error", "message": str(e.detail)},
                    },
                )
            raise
        if protocol == "anthropic":
            collected = chat_completion_to_anthropic_message(collected, original)
        return JSONResponse(content=collected)

    def _stream(
        self, messages: list[dict[str, Any]], model: str, sanitize: bool = True,
        tools: list[dict[str, Any]] | None = None,
        include_usage: bool = False,
        native: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """把 Trae SSE 转成 OpenAI chat.completion.chunk 流。"""
        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        in_thinking = False
        usage: dict[str, Any] | None = None

        def chunk(delta: dict[str, Any], finish: str | None = None) -> str:
            payload = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        def usage_chunk() -> str:
            # OpenAI 流式协议：stream_options.include_usage 时，[DONE] 前须有一个
            # choices 为空、只带 usage 的收尾 chunk。下游客户端（如 ethan）靠它
            # 判定 is_final——缺失会导致 final chunk 永远不到、tool_calls 整体丢失
            payload = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [],
                "usage": usage or {
                    "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                },
            }
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        def error_chunk(msg: str, code: Any = None) -> str:
            # 结构化错误 chunk（与 CodeBuddy 通道 stream_upstream 的错误格式一致）。
            # 不能塞进 content 文本——anthropic 客户端（Claude Code）会把
            # 伪正文当模型输出继续循环；包装层据此转成 event: error
            err: dict[str, Any] = {"message": msg, "type": "upstream_error"}
            if code is not None:
                err["code"] = code
            return f"data: {json.dumps({'error': err}, ensure_ascii=False)}\n\n"

        try:
            # 上游调用放独立读线程：send_trae_chat 是同步整段缓冲读（最长受
            # 上游 socket 180s 超时约束），原实现直接在生成器线程里阻塞读——
            # 生成期间客户端收不到任何字节，长生成（深度 review 大报告等）会
            # 触发下游单 chunk 超时（实测 Ethan _CHUNK_TIMEOUT=120s）→ 回合
            # 中止、落库空回复。拆成读线程 + 带 timeout 的队列等待后，等待
            # 间隙按 TRAE_HEARTBEAT_INTERVAL 发 reasoning 心跳：喂饱下游超时
            # 计时器（续命），也让下游 UI 知道中转还在等上游。
            _ev_q: queue.Queue = queue.Queue()

            def _read_upstream() -> None:
                try:
                    if native is not None:
                        raw_text = _send_native_chat(
                            native["messages"], model, stream=True,
                            tools=native["tools"])
                        used = True
                        if _native_rejected(raw_text):
                            log.warning(
                                "trae native tools rejected (4001), "
                                "fallback to text protocol: model=%s", model)
                            _debug_dump("trae_native_fallback", model=model,
                                        phase="stream")
                            raw_text = send_trae_chat(
                                messages, model, stream=True,
                                base_url=self._base_url)
                            used = False
                        _ev_q.put(("raw", (raw_text, used)))
                    else:
                        _ev_q.put(("raw", (send_trae_chat(
                            messages, model, stream=True,
                            base_url=self._base_url), False)))
                except BaseException as e:  # noqa: BLE001 — 原样转主线程抛出
                    _ev_q.put(("error", e))

            threading.Thread(target=_read_upstream, daemon=True,
                             name="trae-sse-reader").start()
            raw: str | None = None
            used_native = False
            _waited = 0
            while raw is None:
                try:
                    if TRAE_HEARTBEAT_INTERVAL > 0:
                        kind, payload = _ev_q.get(timeout=TRAE_HEARTBEAT_INTERVAL)
                    else:
                        kind, payload = _ev_q.get()  # 心跳关闭：纯阻塞等
                except queue.Empty:
                    _waited += TRAE_HEARTBEAT_INTERVAL
                    _debug_dump("trae_heartbeat", waited=_waited)
                    yield chunk({"reasoning_content": _heartbeat_text(_waited)})
                    continue
                if kind == "error":
                    raise payload
                raw, used_native = payload
            acc = _NativeToolAccumulator() if used_native else None
            cleaner = (
                _StreamLeakCleaner() if sanitize and not used_native else None
            )
            splitter = (
                _StreamToolCallSplitter(_tool_names(tools))
                if tools and not used_native else None
            )
            dbg_parts: list[str] = []
            for event, data in _parse_sse(raw):
                if event == "error":
                    yield error_chunk(_trae_error_text(data), (data or {}).get("code"))
                    yield "data: [DONE]\n\n"
                    return
                if event == "output":
                    if data.get("reasoning_content"):
                        in_thinking = True
                        yield chunk({"reasoning_content": data["reasoning_content"]})
                    if acc is not None:
                        # 原生通道：tool_calls 按结构化事件累积，正文不经
                        # 清洗/分流（无 agent 预设，没有可泄漏的协议语法）
                        if data.get("tool_calls"):
                            acc.feed(data["tool_calls"])
                        text = data.get("response") or ""
                        if text:
                            dbg_parts.append(text)
                            yield chunk({"role": "assistant", "content": text})
                            in_thinking = False
                    elif data.get("response"):
                        text = data["response"]
                        if cleaner is not None:
                            text = cleaner.feed(text)
                        elif splitter is not None:
                            text = splitter.feed(text)
                        if text:
                            dbg_parts.append(text)
                            yield chunk({"role": "assistant", "content": text})
                            in_thinking = False
                elif event == "token_usage":
                    u = data or {}
                    try:
                        usage = {
                            "prompt_tokens": int(u.get("prompt_tokens") or 0),
                            "completion_tokens": int(u.get("completion_tokens") or 0),
                            "total_tokens": int(u.get("total_tokens") or 0),
                        }
                    except (TypeError, ValueError):
                        usage = None
                elif event == "done":
                    break
        except HTTPException as e:
            yield error_chunk(f"trae error {e.status_code}: {e.detail}", e.status_code)
            yield "data: [DONE]\n\n"
            return
        except Exception as e:
            log.error("Trae stream error: %s", e)
            yield error_chunk(f"trae stream error: {e}")
            return

        finish = "stop"
        dbg_calls: list[dict[str, Any]] = []
        if used_native:
            calls = acc.finish()
            if calls:
                finish = "tool_calls"
                dbg_calls = calls
                # OpenAI 流式协议：tool_call 必须带 index（客户端靠它合并分片），
                # 首个 delta 带 role；不带 index 会被部分解析器直接丢弃
                yield chunk({"role": "assistant", "tool_calls": calls})
        elif cleaner is not None:
            tail = cleaner.flush()
            if tail:
                dbg_parts.append(tail)
                yield chunk({"role": "assistant", "content": tail})
        elif splitter is not None:
            tail, calls = splitter.flush()
            if tail:
                dbg_parts.append(tail)
                yield chunk({"role": "assistant", "content": tail})
            if calls:
                finish = "tool_calls"
                dbg_calls = calls
                # OpenAI 流式协议：tool_call 必须带 index（客户端靠它合并分片），
                # 首个 delta 带 role；不带 index 会被部分解析器直接丢弃
                yield chunk({"role": "assistant", "tool_calls": [
                    {"index": i, "id": c["id"], "type": "function",
                     "function": c["function"]}
                    for i, c in enumerate(calls)
                ]})
        _debug_dump("debug_trae_response", model=model, stream=True,
                    content="".join(dbg_parts),
                    tool_calls=[
                        {"name": c["function"]["name"],
                         "arguments": c["function"]["arguments"]}
                        for c in dbg_calls
                    ])
        yield chunk({}, finish)
        if include_usage:
            yield usage_chunk()
        yield "data: [DONE]\n\n"

    def _collect(
        self, messages: list[dict[str, Any]], model: str, sanitize: bool = True,
        tools: list[dict[str, Any]] | None = None,
        native: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """非流式：聚合 Trae SSE 成完整响应。"""
        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        usage_real: dict[str, Any] | None = None

        if native is not None:
            raw = _send_native_chat(
                native["messages"], model, stream=False, tools=native["tools"])
            used_native = True
            if _native_rejected(raw):
                log.warning(
                    "trae native tools rejected (4001), "
                    "fallback to text protocol: model=%s", model)
                _debug_dump("trae_native_fallback", model=model, phase="collect")
                raw = send_trae_chat(messages, model, stream=False,
                                     base_url=self._base_url)
                used_native = False
        else:
            raw = send_trae_chat(messages, model, stream=False,
                                 base_url=self._base_url)
            used_native = False
        acc = _NativeToolAccumulator() if used_native else None
        for event, data in _parse_sse(raw):
            if event == "error":
                raise HTTPException(
                    status_code=502,
                    detail=_trae_error_text(data),
                )
            if event == "token_usage":
                u = data or {}
                try:
                    usage_real = {
                        "prompt_tokens": int(u.get("prompt_tokens") or 0),
                        "completion_tokens": int(u.get("completion_tokens") or 0),
                        "total_tokens": int(u.get("total_tokens") or 0),
                    }
                except (TypeError, ValueError):
                    usage_real = None
            if event == "output":
                if data.get("reasoning_content"):
                    reasoning_parts.append(data["reasoning_content"])
                if data.get("response"):
                    content_parts.append(data["response"])
                if acc is not None and data.get("tool_calls"):
                    acc.feed(data["tool_calls"])

        reasoning = "".join(reasoning_parts)
        content = "".join(content_parts)
        # 无 tools 请求也必须有初始值——下方 _debug_dump 无条件引用
        # （实测缺失时非流式无 tools 请求直接 UnboundLocalError -> internal error）
        tool_calls: list[dict[str, Any]] = []
        if used_native:
            # 原生通道：tool_calls 来自结构化事件；无 agent 预设，正文没有
            # 可泄漏的协议语法，不做解析/清洗
            tool_calls = acc.finish()
        else:
            # 顺序关键：带 tools 的请求必须先解析、后清洗——_sanitize_agent_leak
            # 会把 <tool_call>/</arg_value> 标签全剥掉，先清洗再解析会让解析器
            # 拿到被拆掉结构的残骸（实测 glm-5.3 函数调用表达式因此整段漏进正文）
            if tools:
                content, tool_calls = _parse_tool_calls(content, _tool_names(tools))
            if sanitize:
                content = _sanitize_agent_leak(content)
        _debug_dump("debug_trae_response", model=model, stream=False, content=content,
                    tool_calls=len(tool_calls))
        # 只有 tool_calls 没有正文也是合法响应（agent 直接发起调用），
        # 不注入空响应兜底文案——否则会混进 Anthropic tool_use 消息正文
        if not content and not reasoning and not tool_calls:
            content = "(trae upstream 返回了空响应，未产生任何内容)"
        if reasoning:
            if content:
                content = f"<think>\n{reasoning}\n</think>\n\n{content}"
            else:
                # 只有思维链没有正文时，别把 reasoning 整个丢掉
                content = f"<think>\n{reasoning}\n</think>"

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    **({"tool_calls": tool_calls} if tool_calls else {}),
                },
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }],
            # 原生通道带真实 token_usage；文本协议上游不吐 usage，沿用旧启发式
            "usage": usage_real or {
                "prompt_tokens": 0,
                "completion_tokens": max(1, len(content.encode("utf-8")) // 4),
                "total_tokens": 0,
            },
        }

