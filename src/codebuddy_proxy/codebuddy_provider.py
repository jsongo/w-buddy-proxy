"""CodeBuddy 默认上游 provider 及核心转发逻辑。

包含：
- ``CodeBuddyProvider``：默认上游，实现 ``BaseProvider`` 接口。
- ``forward_chat``：多 provider 路由入口。
- ``stream_upstream`` / ``collect_upstream`` / ``convert_nonstream``：
  流式/非流式转发与协议转换（SSE 解析、DSML、工具调用）。
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Any

import httpx
from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from codebuddy_proxy.dsml_parser import DSMLStreamBuffer
from codebuddy_proxy.providers import BaseProvider
from codebuddy_proxy.state import (
    diagnostic,
    get_state,
    is_policy_blocked,
)
from codebuddy_proxy.logging_setup import now_s

# 可选高级功能模块（脱敏 / 投影 / 协议转换器）
try:
    from codebuddy_proxy.desensitize import desensitize_body
    HAS_DESENSITIZE = True
except ImportError:
    HAS_DESENSITIZE = False

    def desensitize_body(body, **kwargs):
        return body


try:
    from codebuddy_proxy.responses_projection import project_responses_chat_body
    HAS_PROJECTION = True
except ImportError:
    HAS_PROJECTION = False

    def project_responses_chat_body(body):
        return body, {}


try:
    from codebuddy_proxy.responses_adapter import responses_request_to_chat, ResponsesStreamConverter
    HAS_RESPONSES_ADAPTER = True
except ImportError:
    HAS_RESPONSES_ADAPTER = False

    def responses_request_to_chat(body):
        raise RuntimeError("responses_adapter not available - cannot convert /v1/responses requests")

    ResponsesStreamConverter = None


try:
    from codebuddy_proxy.anthropic_adapter import anthropic_to_chat, AnthropicStreamConverter
    HAS_ANTHROPIC_ADAPTER = True
except ImportError:
    HAS_ANTHROPIC_ADAPTER = False

    def anthropic_to_chat(body):
        raise RuntimeError("anthropic_adapter not available - cannot convert /v1/messages requests")

    AnthropicStreamConverter = None


def _normalize_tool_choice(tool_choice: Any) -> Any:
    """把 OpenAI 的 object 形式 tool_choice 转成上游接受的 string 形式。

    上游 CodeBuddy 后端（Go）的 Request.tool_choice 字段是 string 类型，
    OpenAI 标准里 ``{"type":"function","function":{"name":"X"}}`` 这种 object
    形式（强制调用函数 X）会触发 400：cannot unmarshal object into ...
    of type string。此处转换为等价的函数名字符串 "X"（实测上游接受且语义一致）。
    """
    if isinstance(tool_choice, dict):
        name = (tool_choice.get("function") or {}).get("name")
        if name:
            return name
        # {"type": "function"} 但缺 name：退化为 required（强制调用工具）
        if tool_choice.get("type") == "function":
            return "required"
    return tool_choice


class CodeBuddyProvider(BaseProvider):
    """默认的 CodeBuddy 上游，实现 BaseProvider 接口。

    与豆包（DoubaoProvider）对称统一。CodeBuddy 的复杂转发逻辑
    （SSE 解析、DSML、工具调用、协议转换）仍由本模块的
    ``stream_upstream`` / ``collect_upstream`` / ``convert_nonstream``
    承担，本类只做「认证 + 构造上游请求 + 分派流式/非流式」。
    """

    id = "codebuddy"
    name = "CodeBuddy"

    def models(self) -> list[dict[str, Any]]:
        # CodeBuddy 的模型列表由 /v1/models 统一从本地配置加载，
        # 此处返回空（不参与 provider 路由的模型合并，避免重复）。
        return []

    def ensure_auth(self) -> None:
        state = get_state()
        state.ensure_auth()

    async def forward(
        self,
        body: dict[str, Any],
        protocol: str,
        original: dict[str, Any] | None = None,
    ) -> StreamingResponse | JSONResponse:
        state = get_state()
        state.ensure_auth()

        diagnostic("upstream_request", protocol=protocol, **body_summary(body))

        stream = bool(body.get("stream"))
        upstream_body = dict(body)

        # 归一化 tool_choice：object 形式 → 函数名字符串（上游只接受 string）
        if "tool_choice" in upstream_body:
            upstream_body["tool_choice"] = _normalize_tool_choice(upstream_body["tool_choice"])

        # 应用脱敏处理
        if state.enable_desensitize:
            upstream_body = desensitize_body(upstream_body, compact_harness=True)

        # 始终以流式方式请求上游（聚合或转发）
        upstream_body["stream"] = True
        upstream_body.setdefault("stream_options", {"include_usage": True})

        url = state.client.endpoint + "/v2/chat/completions"
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; Genie-IDE/1.0)",
            **state.client.auth_headers(),
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        # 🔍 调试：输出实际发送的IDE识别headers
        if state.logger:
            ide_headers = {k: v for k, v in headers.items()
                           if k.startswith("X-IDE-") or k == "X-Product-Version" or k == "X-Machine-Id"}
            diagnostic("upstream_ide_headers", **ide_headers)

        if stream:
            # 流式：直接转发
            return StreamingResponse(
                stream_upstream(url, headers, upstream_body, protocol, original),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "close"},
            )
        else:
            # 非流式：聚合后返回
            collected = await collect_upstream(url, headers, upstream_body, protocol)
            return JSONResponse(content=convert_nonstream(collected, protocol, original))


async def forward_chat(
    body: dict[str, Any],
    protocol: str,
    original: dict[str, Any] | None = None,
) -> StreamingResponse | JSONResponse:
    """转发 chat 请求到上游，支持流式和非流式。

    多 provider 路由：若请求模型命中某个非默认 provider（如豆包），
    走该 provider 的 forward；否则走默认 CodeBuddy。
    """
    state = get_state()

    # ---- 多 provider 路由 ----
    requested_model = body.get("model")
    providers = getattr(state, "providers", {}) or {}
    provider = None
    for p in providers.values():
        if any(m.get("id") == requested_model for m in p.models()):
            provider = p
            break
    if provider is not None:
        # 非默认 provider（豆包等）：仅 openai 协议透传（doubao2api 只支持
        # OpenAI chat completions；responses/anthropic 协议由调用方决定，
        # 这里统一按 openai 透传，客户端应使用 openai 协议接入）。
        diagnostic("provider_route", provider=provider.id, model=requested_model, protocol=protocol)
        try:
            provider.ensure_auth()
        except HTTPException:
            raise
        return await provider.forward(body, protocol, original)

    # 默认 CodeBuddy 路径（对称封装，与其它 provider 一致）
    return await _default_codebuddy.forward(body, protocol, original)


# 默认 CodeBuddy provider 单例（供 forward_chat 默认路径调用）。
# 定义在 CodeBuddyProvider 类之后，实例化安全。
_default_codebuddy = CodeBuddyProvider()


def _build_openai_flush_chunk(
    residual: str, last_chunk_id: str, last_chunk_created: int, last_chunk_model: str
) -> dict:
    """构造流结束 flush 时发出的 OpenAI ChatCompletionChunk。

    必须补全顶层 id/object/created/model 字段，否则严格解析的客户端
    （如 grok）会报 ``serialization error: missing field id``。
    """
    return {
        "id": last_chunk_id,
        "object": "chat.completion.chunk",
        "created": last_chunk_created,
        "model": last_chunk_model,
        "choices": [{
            "index": 0,
            "delta": {"content": residual},
        }],
    }


async def stream_upstream(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    protocol: str,
    original: dict[str, Any] | None,
):
    """异步流式转发上游响应到客户端。

    关键改进：
    1. 使用 httpx.AsyncClient 异步请求
    2. aiter_lines() 自动处理行分割和超时
    3. 记录流开始/进度/完成日志
    """
    state = get_state()
    stream_start_time = time.time()

    # 【日志】流开始
    state.write_log("stream_started", protocol=protocol, timestamp=stream_start_time)
    diagnostic("stream_started", protocol=protocol)

    response_id = "resp_" + uuid.uuid4().hex
    anthropic_state = AnthropicStreamConverter(
        (original or {}).get("model", "default")
    ) if protocol == "anthropic" and AnthropicStreamConverter else None

    # Responses 协议转换器（使用传入的 body 参数）
    responses_state = ResponsesStreamConverter(
        model=body.get("model", "auto")
    ) if protocol == "responses" and ResponsesStreamConverter else None

    # DSML 缓冲区（用于处理可能的文本标记格式工具调用）
    dsml_buffer = DSMLStreamBuffer()

    # 【修复 B1】原生流式 tool_calls name 缓存
    # 上游首 chunk 带完整 name，后续 chunk name 为空但有 arguments 分片
    # 维护 name 缓存防止空值覆盖
    native_tool_name_by_index: dict[int, str] = {}

    emitted_response_created = False
    response_text = ""
    response_text_started = False
    chunk_count = 0
    done_seen = False
    raw_chunks: list[bytes] = []
    last_progress_log = stream_start_time
    detected_tool_calls = []
    # 【修复 C3】记录最后一个上游 chunk 的元数据，供流结束 flush 时复用，
    # 保证 final_chunk 补全 OpenAI ChatCompletionChunk 必需的顶层字段
    last_chunk_id: str = ""
    last_chunk_created: int = 0
    last_chunk_model: str = ""
    try:
        # 异步HTTP客户端：timeout=None 依赖TCP超时
        # 使用合理的超时配置：连接超时30s，读取超时300s
        timeout_config = httpx.Timeout(30.0, read=300.0)
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            async with client.stream("POST", url, headers=headers, json=body) as resp:
                if resp.status_code != 200:
                    error_body = await resp.aread()
                    error_text = error_body.decode("utf-8", "replace")

                    # 【诊断】记录 400 错误时的工具定义
                    if resp.status_code == 400:
                        tools = body.get("tools", [])
                        diagnostic("upstream_400_error",
                                   status=resp.status_code,
                                   error_preview=error_text[:200],
                                   tool_count=len(tools),
                                   sample_tools=tools[:2] if tools else [])
                    diagnostic("upstream_error", protocol=protocol,
                        status=resp.status_code,
                        detail=error_text[:500])

                    # 返回结构化错误（包含详细信息）
                    if protocol == "anthropic":
                        # Anthropic error format
                        error_event = {
                            "type": "error",
                            "error": {
                                "type": "api_error",
                                "message": f"Upstream API error (HTTP {resp.status_code}): {error_text[:200]}"
                            }
                        }
                        yield f"event: error\ndata: {json.dumps(error_event, ensure_ascii=False)}\n\n".encode()
                    else:
                        # OpenAI error format
                        error_chunk = {
                            "error": {
                                "message": f"Upstream API error (HTTP {resp.status_code})",
                                "type": "upstream_error",
                                "code": resp.status_code,
                                "details": error_text[:500]
                            }
                        }
                        yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n".encode()

                    return

                diagnostic("upstream_response", protocol=protocol, status=resp.status_code)

                # 【配置】超时策略 - 双重超时机制
                # 1. 空闲超时（idle timeout）：连续N秒没有新chunk → 超时
                # 2. 总时长上限（total duration）：绝对时长限制，防止无限期占用
                client_timeout = body.get("max_stream_duration")
                client_idle_timeout = body.get("max_idle_duration")

                # 空闲超时：默认60秒，客户端可配置（10-300秒）
                if client_idle_timeout is not None:
                    MAX_IDLE_DURATION = min(max(int(client_idle_timeout), 10), 300)
                else:
                    MAX_IDLE_DURATION = 60  # 60秒没有新数据则超时

                # 总时长上限：默认30分钟，客户端可配置（最大2小时）
                if client_timeout is not None:
                    MAX_TOTAL_DURATION = min(max(int(client_timeout), 60), 7200)
                else:
                    MAX_TOTAL_DURATION = 1800  # 30分钟绝对上限

                last_chunk_time = time.time()  # 记录最后一次收到chunk的时间

                # 异步迭代行（自动处理超时和分块）
                async for line in resp.aiter_lines():
                    current_time = time.time()

                    # 【保护1】检查空闲超时：距离上次chunk超过N秒
                    idle_time = current_time - last_chunk_time
                    if idle_time > MAX_IDLE_DURATION:
                        diagnostic("stream_idle_timeout", protocol=protocol,
                                  chunks=chunk_count,
                                  idle_time=round(idle_time, 2),
                                  max_idle=MAX_IDLE_DURATION)
                        state.write_log("stream_idle_timeout", protocol=protocol,
                                       chunks=chunk_count, idle_time=round(idle_time, 2))
                        break  # 空闲超时，结束流

                    # 【保护2】检查总时长上限：防止无限期运行
                    total_elapsed = current_time - stream_start_time
                    if total_elapsed > MAX_TOTAL_DURATION:
                        diagnostic("stream_total_duration_exceeded", protocol=protocol,
                                  chunks=chunk_count,
                                  elapsed=round(total_elapsed, 2),
                                  max_duration=MAX_TOTAL_DURATION)
                        state.write_log("stream_total_duration_exceeded", protocol=protocol,
                                       chunks=chunk_count, elapsed=round(total_elapsed, 2))
                        break  # 总时长超限，结束流

                    # 更新最后chunk时间
                    last_chunk_time = current_time

                    # 【日志】进度记录（每10个chunk且间隔5秒）
                    if chunk_count > 0 and chunk_count % 10 == 0:
                        now = time.time()
                        if now - last_progress_log >= 5:
                            diagnostic("stream_progress", protocol=protocol,
                                chunks=chunk_count,
                                elapsed=round(now - stream_start_time, 2))
                            last_progress_log = now
                    line = line.strip()
                    raw_chunks.append(line.encode("utf-8"))

                    if not line.startswith("data:"):
                        continue

                    data = line[5:].strip()
                    if data == "[DONE]":
                        done_seen = True
                        break

                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    chunk_count += 1

                    # 【修复 C3】记录最后一个上游 chunk 的元数据，供流结束 flush 复用
                    if chunk.get("id"):
                        last_chunk_id = chunk["id"]
                    if chunk.get("created"):
                        last_chunk_created = chunk["created"]
                    if chunk.get("model"):
                        last_chunk_model = chunk["model"]

                    # 提取当前 chunk 的原生 tool_calls（三种协议共享）。
                    # 注意：必须在协议分支之前定义，responses/anthropic 分支
                    # 的 B2 覆盖条件也会引用它（原生 tool_calls 存在时不覆盖）。
                    native_tool_calls = (
                        ((chunk.get("choices") or [{}])[0].get("delta") or {}).get("tool_calls")
                    )

                    # 根据协议转换事件
                    if protocol == "openai":
                        # 【修复 Bug B3】删除空 finish_reason 字段
                        # 上游可能返回 "finish_reason": "" 或 null，导致客户端序列化失败
                        # 完全删除该字段，只保留真实的 stop/tool_calls/length/content_filter
                        if "choices" in chunk:
                            for choice in chunk["choices"]:
                                if "finish_reason" in choice and (choice["finish_reason"] == "" or choice["finish_reason"] is None):
                                    del choice["finish_reason"]

                        # 【修复 Bug 2】原生流式 tool_calls name 缓存
                        # 首 chunk 带完整 name，后续 chunk name 为空但带 arguments 分片，
                        # 这里按 index 缓存 name 并回填（native_tool_calls 已在协议分支前提取）
                        if native_tool_calls:
                            for tc in native_tool_calls:
                                idx = tc.get("index", 0)
                                fn = tc.get("function") or {}
                                nm = fn.get("name") or ""

                                # 首次出现非空 name：记录到缓存
                                if nm:
                                    native_tool_name_by_index[idx] = nm
                                # 后续空 name：从缓存回填
                                elif idx in native_tool_name_by_index:
                                    if "function" not in tc:
                                        tc["function"] = {}
                                    tc["function"]["name"] = native_tool_name_by_index[idx]

                        # 提取 content 并通过 DSML 缓冲区处理
                        chunk_content = str(
                            ((chunk.get("choices") or [{}])[0].get("delta") or {}).get("content") or ""
                        )

                        if chunk_content:
                            # 使用 DSML 缓冲区处理（清理标记，检测工具调用）
                            cleaned_content, chunk_tool_calls = dsml_buffer.add_chunk(chunk_content)

                            # 累积清理后的文本
                            if cleaned_content:
                                response_text += cleaned_content

                            # 记录检测到的工具调用
                            if chunk_tool_calls:
                                detected_tool_calls.extend(chunk_tool_calls)

                            # 修改 chunk 中的 content 为清理后的内容
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                if "delta" not in chunk["choices"][0]:
                                    chunk["choices"][0]["delta"] = {}
                                chunk["choices"][0]["delta"]["content"] = cleaned_content

                        # 【修复 Bug B2】仅当 chunk 不含原生 tool_calls 时，才使用 DSML 解析的工具调用
                        # DSML 用于兜底：处理上游以文本标记返回工具调用的场景
                        # 如果 chunk 已有原生 delta.tool_calls，原样透传，绝不覆盖
                        if detected_tool_calls and dsml_buffer.should_emit_tool_calls() and not native_tool_calls:
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                chunk["choices"][0]["finish_reason"] = "tool_calls"
                                # 将检测到的工具调用转换为 OpenAI 格式
                                chunk["choices"][0]["delta"]["tool_calls"] = [
                                    {
                                        "index": idx,
                                        "id": f"call_{uuid.uuid4().hex[:24]}",
                                        "type": "function",
                                        "function": {
                                            "name": tc["name"],
                                            "arguments": json.dumps(tc["input"], ensure_ascii=False)
                                        }
                                    }
                                    for idx, tc in enumerate(detected_tool_calls)
                                ]

                        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode()

                    elif protocol == "responses" and responses_state:
                        # 【修复】提取 content 并通过 DSML 缓冲区处理
                        chunk_content = str(
                            ((chunk.get("choices") or [{}])[0].get("delta") or {}).get("content") or ""
                        )

                        if chunk_content:
                            # 使用 DSML 缓冲区处理（清理标记，检测工具调用）
                            cleaned_content, chunk_tool_calls = dsml_buffer.add_chunk(chunk_content)

                            # 累积清理后的文本
                            if cleaned_content:
                                response_text += cleaned_content

                            # 记录检测到的工具调用
                            if chunk_tool_calls:
                                detected_tool_calls.extend(chunk_tool_calls)

                            # 修改 chunk 中的 content 为清理后的内容
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                if "delta" not in chunk["choices"][0]:
                                    chunk["choices"][0]["delta"] = {}
                                chunk["choices"][0]["delta"]["content"] = cleaned_content

                            # 【修复 Bug B2】仅当 chunk 不含原生 tool_calls 时，才使用 DSML 解析的工具调用
                            if chunk_tool_calls and dsml_buffer.should_emit_tool_calls() and not native_tool_calls:
                                if "choices" in chunk and len(chunk["choices"]) > 0:
                                    chunk["choices"][0]["finish_reason"] = "tool_calls"
                                    chunk["choices"][0]["delta"]["tool_calls"] = [
                                        {
                                            "index": idx,
                                            "id": f"call_{uuid.uuid4().hex[:24]}",
                                            "type": "function",
                                            "function": {
                                                "name": tc["name"],
                                                "arguments": json.dumps(tc["input"], ensure_ascii=False)
                                            }
                                        }
                                        for idx, tc in enumerate(detected_tool_calls)
                                    ]

                        # 使用 ResponsesStreamConverter 转换事件（此时 chunk 已经被清理）
                        events = responses_state.feed_chunk(chunk)
                        for event_name, event_data in events:
                            yield f"event: {event_name}\ndata: {json.dumps(event_data, ensure_ascii=False)}\n\n".encode()
                    elif protocol == "anthropic" and anthropic_state:
                        # 提取 content 并通过 DSML 缓冲区处理
                        chunk_content = str(
                            ((chunk.get("choices") or [{}])[0].get("delta") or {}).get("content") or ""
                        )

                        if chunk_content:
                            # 使用 DSML 缓冲区处理（清理标记，检测工具调用）
                            cleaned_content, chunk_tool_calls = dsml_buffer.add_chunk(chunk_content)

                            # 累积清理后的文本
                            if cleaned_content:
                                response_text += cleaned_content

                            # 记录检测到的工具调用
                            if chunk_tool_calls:
                                detected_tool_calls.extend(chunk_tool_calls)

                            # ✅ 关键修复：在传递给 AnthropicStreamConverter 之前，先修改 chunk
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                if "delta" not in chunk["choices"][0]:
                                    chunk["choices"][0]["delta"] = {}
                                # 使用清理后的内容替换原始内容
                                chunk["choices"][0]["delta"]["content"] = cleaned_content

                            # 【修复 Bug B2】仅当 chunk 不含原生 tool_calls 时，才使用 DSML 解析的工具调用
                            if chunk_tool_calls and dsml_buffer.should_emit_tool_calls() and not native_tool_calls:
                                if "choices" in chunk and len(chunk["choices"]) > 0:
                                    chunk["choices"][0]["finish_reason"] = "tool_calls"
                                    chunk["choices"][0]["delta"]["tool_calls"] = [
                                        {
                                            "index": idx,
                                            "id": call.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                                            "type": "function",
                                            "function": {
                                                "name": call["function"]["name"],
                                                "arguments": call["function"]["arguments"]
                                            }
                                        }
                                        for idx, call in enumerate(chunk_tool_calls)
                                    ]

                        # 转换为 Anthropic 事件（此时 chunk 已经被清理）
                        events = anthropic_state.feed_chunk(chunk)
                        for event_name, event_data in events:
                            yield f"event: {event_name}\ndata: {json.dumps(event_data, ensure_ascii=False)}\n\n".encode()

                # 发送结束事件
                if protocol == "responses" and responses_state:
                    # 【修复】流结束前强制刷新 DSML 缓冲区残留内容
                    residual = dsml_buffer.flush()
                    if residual:
                        for event_name, event_data in responses_state.feed_chunk({
                            "choices": [{"index": 0, "delta": {"content": residual}}]
                        }):
                            yield f"event: {event_name}\ndata: {json.dumps(event_data, ensure_ascii=False)}\n\n".encode()
                    # 使用 ResponsesStreamConverter 的 finish() 方法发出完整事件序列
                    for event_name, event_data in responses_state.finish():
                        yield f"event: {event_name}\ndata: {json.dumps(event_data, ensure_ascii=False)}\n\n".encode()

                    # 【修复】发送SSE流结束标记，防止客户端持续等待
                    yield b"data: [DONE]\n\n"

                elif protocol == "anthropic" and anthropic_state:
                    # 【修复】流结束前强制刷新 DSML 缓冲区残留内容
                    residual = dsml_buffer.flush()
                    if residual:
                        for event_name, event_data in anthropic_state.feed_chunk({
                            "choices": [{"index": 0, "delta": {"content": residual}}]
                        }):
                            yield f"event: {event_name}\ndata: {json.dumps(event_data, ensure_ascii=False)}\n\n".encode()
                    for event_name, event_data in anthropic_state.finish():
                        yield f"event: {event_name}\ndata: {json.dumps(event_data, ensure_ascii=False)}\n\n".encode()

                    # 【修复】发送SSE流结束标记，防止客户端持续等待（与Responses协议保持一致）
                    # 虽然Anthropic有message_stop事件，但明确的[DONE]标记能确保客户端立即处理最后的内容
                    yield b"data: [DONE]\n\n"

                elif protocol == "openai":
                    # 【修复】流结束前强制刷新 DSML 缓冲区残留内容，避免含 '<' 的
                    # 普通文本（如文档中举例的工具调用标签）被永久扣留而截断输出
                    residual = dsml_buffer.flush()
                    if residual:
                        response_text += residual
                        final_chunk = _build_openai_flush_chunk(
                            residual, last_chunk_id, last_chunk_created, last_chunk_model
                        )
                        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n".encode()
                    yield b"data: [DONE]\n\n"

    except httpx.TimeoutException as exc:
        # 【日志】超时
        diagnostic("stream_timeout", protocol=protocol, chunks=chunk_count,
            elapsed=round(time.time() - stream_start_time, 2), error=str(exc))
        state.write_log("stream_timeout", protocol=protocol, chunks=chunk_count, error=str(exc))

        # 【防御性编程】确保发出完整的事件序列
        if protocol == "responses" and responses_state:
            for event_name, event_data in responses_state.finish():
                yield f"event: {event_name}\ndata: {json.dumps(event_data, ensure_ascii=False)}\n\n".encode()
        elif protocol == "anthropic" and anthropic_state:
            for event_name, event_data in anthropic_state.finish():
                yield f"event: {event_name}\ndata: {json.dumps(event_data, ensure_ascii=False)}\n\n".encode()

        # 发送错误事件
        error_chunk = {
            "error": {
                "message": f"stream timeout after {chunk_count} chunks",
                "type": "timeout_error"
            }
        }
        yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n".encode()

    except Exception as exc:
        # 【日志】其他错误
        diagnostic("stream_error", protocol=protocol, chunks=chunk_count,
            elapsed=round(time.time() - stream_start_time, 2), error=str(exc))
        state.write_log("stream_error", protocol=protocol, chunks=chunk_count, error=str(exc))

        # 【防御性编程】确保发出完整的事件序列
        if protocol == "responses" and responses_state:
            for event_name, event_data in responses_state.finish():
                yield f"event: {event_name}\ndata: {json.dumps(event_data, ensure_ascii=False)}\n\n".encode()
        elif protocol == "anthropic" and anthropic_state:
            for event_name, event_data in anthropic_state.finish():
                yield f"event: {event_name}\ndata: {json.dumps(event_data, ensure_ascii=False)}\n\n".encode()

        # 发送错误事件
        error_chunk = {
            "error": {
                "message": f"stream error: {exc}",
                "type": "internal_error"
            }
        }
        yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n".encode()

    finally:
        # 【日志】流完成
        if state.verbose_llm:
            raw_response = b"\n".join(raw_chunks)
            state.write_body_log("upstream_response", raw_response, protocol=protocol,
                                status=200, method="POST", path="/v2/chat/completions")

        logged_text = (
            anthropic_state.text if anthropic_state
            else responses_state.text if responses_state
            else response_text
        )
        stream_duration = round(time.time() - stream_start_time, 2)

        log_upstream_response(protocol, logged_text, stream=True,
                            chunk_count=chunk_count, duration=stream_duration,
                            upstream_done=done_seen)


async def collect_upstream(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    protocol: str,
) -> dict[str, Any]:
    """聚合上游流式响应为单个 JSON 对象（非流式场景）。"""
    state = get_state()

    usage = None
    finish_reason = None
    content = ""
    tool_calls_dict: dict[int, dict] = {}  # 使用 dict 按 index 累加
    # DSML 缓冲区
    dsml_buffer = DSMLStreamBuffer()

    try:
        # 使用合理的超时配置：连接超时30s，读取超时300s
        timeout_config = httpx.Timeout(30.0, read=300.0)
        async with httpx.AsyncClient(timeout=timeout_config) as client:
            async with client.stream("POST", url, headers=headers, json=body) as resp:
                if resp.status_code != 200:
                    error_body = await resp.aread()
                    raise HTTPException(
                        status_code=resp.status_code,
                        detail={"error": {"message": error_body.decode("utf-8", "replace")[:500], "type": "upstream_error"}}
                    )

                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue

                    data = line[5:].strip()
                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    usage = chunk.get("usage") or usage

                    for choice in chunk.get("choices") or []:
                        finish_reason = choice.get("finish_reason") or finish_reason
                        delta = choice.get("delta") or {}

                        # 处理 content（可能包含 DSML）
                        if delta.get("content"):
                            chunk_content = delta["content"]

                            # 使用 DSML 缓冲区处理
                            cleaned_content, detected_tool_calls = dsml_buffer.add_chunk(chunk_content)

                            # 调试日志：记录 DSML 解析结果
                            if detected_tool_calls:
                                diagnostic("dsml_detected",
                                          tool_count=len(detected_tool_calls),
                                          tools=[tc["function"]["name"] for tc in detected_tool_calls])

                            # 累积清理后的 content
                            if cleaned_content:
                                content += cleaned_content

                            # 如果检测到 tool_calls，添加到 dict 中
                            if detected_tool_calls:
                                for detected_call in detected_tool_calls:
                                    # 找到下一个可用的 index
                                    next_idx = len(tool_calls_dict)
                                    tool_calls_dict[next_idx] = detected_call

                        # 处理原生 tool_calls（使用 dict 累加，避免预填充）
                        if delta.get("tool_calls"):
                            for tc in delta["tool_calls"]:
                                idx = tc.get("index", 0)
                                if idx not in tool_calls_dict:
                                    tool_calls_dict[idx] = {
                                        "id": "",
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""}
                                    }

                                if tc.get("id"):
                                    tool_calls_dict[idx]["id"] = tc["id"]
                                if tc.get("function", {}).get("name"):
                                    tool_calls_dict[idx]["function"]["name"] = tc["function"]["name"]
                                if tc.get("function", {}).get("arguments"):
                                    tool_calls_dict[idx]["function"]["arguments"] += tc["function"]["arguments"]

    except httpx.HTTPError as exc:
        diagnostic("upstream_error", protocol=protocol, error=str(exc))
        raise HTTPException(status_code=502, detail={"error": {"message": f"upstream error: {exc}", "type": "upstream_error"}})

    # 转换为 list 并过滤掉无效的 tool_calls（name 为空的）
    tool_calls = [
        v for k, v in sorted(tool_calls_dict.items())
        if v["function"]["name"]
    ]

    # 如果检测到 DSML tool_calls，修改 finish_reason
    if tool_calls and dsml_buffer.should_emit_tool_calls():
        finish_reason = "tool_calls"

    # 【日志】收集完成
    if state.verbose_llm:
        # collect_upstream 没有保存原始响应，只记录聚合后的内容
        pass

    log_upstream_response(protocol, content, stream=False)
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex,
        "object": "chat.completion",
        "created": now_s(),
        "model": body.get("model", "auto"),
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content,
                "tool_calls": tool_calls if tool_calls else None
            },
            "finish_reason": finish_reason or "stop"
        }],
        "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    }


def convert_nonstream(data: dict[str, Any], protocol: str, original: dict[str, Any] | None) -> dict[str, Any]:
    """将聚合的 OpenAI 格式转换为目标协议格式。"""
    if protocol == "openai":
        return data

    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content", "")

    if protocol == "anthropic":
        content_blocks = []
        if content:
            content_blocks.append({"type": "text", "text": content})
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            try:
                arguments = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                arguments = fn.get("arguments", "")
            content_blocks.append({
                "type": "tool_use",
                "id": call.get("id", ""),
                "name": fn.get("name", ""),
                "input": arguments
            })
        return {
            "id": "msg_" + uuid.uuid4().hex,
            "type": "message",
            "role": "assistant",
            "model": (original or {}).get("model", data.get("model", "default")),
            "content": content_blocks,
            "stop_reason": "tool_use" if message.get("tool_calls") else "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": (data.get("usage") or {}).get("prompt_tokens", 0),
                "output_tokens": (data.get("usage") or {}).get("completion_tokens", 0)
            }
        }

    elif protocol == "responses":
        # 构建 content 数组（文本 + 工具调用）
        content_parts = []
        if content:
            content_parts.append({"type": "output_text", "text": content})

        # 处理工具调用
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            try:
                arguments = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                arguments = fn.get("arguments", "")

            content_parts.append({
                "type": "function_call",
                "id": call.get("id", ""),
                "name": fn.get("name", ""),
                "arguments": arguments
            })

        return {
            "id": "resp_" + uuid.uuid4().hex,
            "object": "response",
            "created_at": now_s(),
            "status": "completed",
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": content_parts
            }]
        }

    return data


# ============================================================================
# 日志辅助函数（body_summary 等，供 routes 与转发逻辑共用）
# ============================================================================

def body_summary(body: dict[str, Any]) -> dict[str, Any]:
    messages = body.get("messages") or []
    message_summary = []
    for item in messages:
        if not isinstance(item, dict):
            continue
        content = item.get("content", "")
        if isinstance(content, str):
            content_length = len(content)
            content_type = "text"
        elif isinstance(content, list):
            content_length = sum(
                len(str(part.get("text", ""))) for part in content if isinstance(part, dict)
            )
            content_type = "parts"
        else:
            content_length = 0
            content_type = type(content).__name__
        message_summary.append({
            "role": item.get("role"),
            "content_type": content_type,
            "content_length": content_length,
        })
    return {
        "model": body.get("model"),
        "stream": bool(body.get("stream")),
        "message_count": len(messages),
        "messages": message_summary,
        "tool_count": len(body.get("tools") or []),
    }


def log_client_request(method: str, path: str, body: dict[str, Any] | None) -> None:
    """Log client request with verbosity control."""
    state = get_state()

    if state.verbose_llm:
        state.write_log("client_request", method=method, path=path, body=body)
    else:
        if body:
            summary = body_summary(body)
            state.write_log("client_request_summary", method=method, path=path, **summary)
        else:
            state.write_log("client_request_summary", method=method, path=path)


def log_upstream_request(protocol: str, body: dict[str, Any]) -> None:
    """Log upstream request with verbosity control."""
    state = get_state()

    if state.verbose_llm:
        state.write_log("upstream_request", protocol=protocol,
                       method="POST", path="/v2/chat/completions", body=body)
        diagnostic("upstream_request", protocol=protocol, **body_summary(body))
    else:
        messages = body.get("messages", [])
        total_chars = sum(
            len(str(m.get("content", "")))
            for m in messages
            if isinstance(m, dict)
        )
        summary = {
            "model": body.get("model"),
            "message_count": len(messages),
            "tool_count": len(body.get("tools", [])),
            "stream": bool(body.get("stream")),
            "total_chars": total_chars
        }
        state.write_log("upstream_request_summary", protocol=protocol, **summary)
        diagnostic("upstream_request_summary", protocol=protocol, **summary)


def log_upstream_response(protocol: str, text: str, **stats) -> None:
    """Log upstream response with verbosity control."""
    state = get_state()

    common = {
        "protocol": protocol,
        "content_length": len(text),
        "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
        "safety_message_detected": is_policy_blocked(text),
        **stats
    }

    if state.verbose_llm:
        common["content_preview"] = text[:200] if text else ""

    diagnostic("response", **common)
    state.write_log("stream_completed" if stats.get("stream") else "response",
                   **{k: v for k, v in common.items()
                      if k not in ("content_preview",)})
