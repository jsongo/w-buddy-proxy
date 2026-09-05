"""Trae provider anthropic 协议支持测试（完全离线，不访问 Trae 上游）。

覆盖 /v1/messages（Claude Code 等 Anthropic 客户端）→ Trae 通道的完整链路：
- 响应侧（本次新增能力的核心）：
  * 非流式：chat_completion_to_anthropic_message（thinking / text / tool_use / usage）
  * 流式：_wrap_anthropic_stream（message_start → thinking/text/tool_use 块 → message_stop）
  * 集成：/v1/messages 路由到 trae provider 的端到端（流式 + 非流式 + 工具调用）
- 请求侧回归：system / tools / tool_result 经 anthropic_to_chat 正确转成 chat 请求

运行：
    .venv/bin/python -m pytest test_trae_anthropic.py -v
"""
from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from buddy_proxy import __main__ as m
from buddy_proxy import state as st
from buddy_proxy import trae_provider as tp
from buddy_proxy.anthropic_adapter import (
    AnthropicStreamConverter,
    chat_completion_to_anthropic_message,
)


# ---------------------------------------------------------------------------
# Trae 上游 SSE fixture（send_trae_chat 的返回值形态）
# ---------------------------------------------------------------------------
def _trae_sse(outputs: list[dict]) -> str:
    """构造 Trae 上游原始 SSE 文本（event: output / token_usage / done）。"""
    parts = []
    for o in outputs:
        parts.append("event: output")
        parts.append("data: " + json.dumps(o, ensure_ascii=False))
        parts.append("")
    parts += [
        "event: token_usage",
        'data: {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}',
        "",
        "event: done",
        "data: {}",
        "",
        "",
    ]
    return "\n".join(parts)


SSE_PLAIN = _trae_sse([
    {"reasoning_content": "让我想想"},
    {"response": "你好！"},
])

SSE_TOOL_CALL = _trae_sse([
    {"response": '<tool_call>\n{"name": "get_weather", "arguments": {"city": "北京"}}\n</tool_call>'},
])


# ---------------------------------------------------------------------------
# fixture：假 ProxyState（含 trae provider）+ mock Trae 上游
# ---------------------------------------------------------------------------
@pytest.fixture()
def trae_env(monkeypatch):
    """注册 TraeProvider 并 mock 掉认证与上游调用，返回 (state, calls)。"""
    provider = tp.TraeProvider()
    state = SimpleNamespace(
        client=mock.MagicMock(),
        providers={"trae": provider},
        default_provider="codebuddy",
        mock_dir=None,
        started_at=time.time(),
        enable_desensitize=False,
        enable_optimize_context=False,
        verbose_llm=False,
        write_log=mock.MagicMock(),
        ensure_auth=mock.MagicMock(),
        logger=mock.MagicMock(),
        json_logger=mock.MagicMock(),
        runtime_info={"app_version": "test"},
    )
    monkeypatch.setattr(st, "proxy_state", state)

    calls: list[dict] = []

    def fake_send(messages, model, stream, base_url=None):
        calls.append({"messages": messages, "model": model, "stream": stream})
        if getattr(state, "_upstream_sse", None):
            return state._upstream_sse
        return SSE_TOOL_CALL if state._tool_mode else SSE_PLAIN

    state._tool_mode = False
    # 拆分后 send_trae_chat/_auth 在 trae.provider 命名空间内被调用，
    # monkeypatch 必须 patch 使用处所在模块（mock 语义），shim 不再是 lookup 点
    from buddy_proxy.trae import provider as _trae_provider_impl
    monkeypatch.setattr(_trae_provider_impl, "send_trae_chat", fake_send)
    monkeypatch.setattr(_trae_provider_impl, "_auth", lambda: ("fake-token", "fake-uid"))
    # 本文件验证文本协议路径（教学 + 文本解析）；原生 function calling 通道
    # 的离线测试见 test_trae_native_tools.py
    monkeypatch.setattr(_trae_provider_impl, "_NATIVE_TOOLS_ENABLED", False)
    return state, calls


@pytest.fixture()
def client(trae_env):
    return TestClient(m.app)


def _parse_sse_events(text: str) -> list[tuple[str, dict | str]]:
    """解析代理返回的 SSE 文本 → [(event_name, data_dict | '[DONE]'), ...]。"""
    events = []
    current_event = ""
    current_data: list[str] = []

    def flush():
        if current_data:
            data_str = "\n".join(current_data)
            if data_str == "[DONE]":
                events.append((current_event, "[DONE]"))
            else:
                events.append((current_event, json.loads(data_str)))
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


# ---------------------------------------------------------------------------
# 单元测试：chat_completion_to_anthropic_message（非流式转换）
# ---------------------------------------------------------------------------
def test_nonstream_convert_text_only():
    data = {
        "choices": [{"message": {"role": "assistant", "content": "hello"}}],
        "model": "glm-5.2",
        "usage": {"prompt_tokens": 5, "completion_tokens": 7},
    }
    msg = chat_completion_to_anthropic_message(data, {"model": "claude-x"})
    assert msg["type"] == "message"
    assert msg["role"] == "assistant"
    assert msg["model"] == "claude-x"  # original 优先
    assert msg["content"] == [{"type": "text", "text": "hello"}]
    assert msg["stop_reason"] == "end_turn"
    assert msg["usage"] == {"input_tokens": 5, "output_tokens": 7}


def test_nonstream_convert_think_split():
    data = {
        "choices": [{"message": {"role": "assistant",
                                 "content": "<think>\n推理过程\n</think>\n\n结论"}}],
    }
    msg = chat_completion_to_anthropic_message(data)
    assert msg["content"][0]["type"] == "thinking"
    assert msg["content"][0]["thinking"] == "推理过程"
    assert msg["content"][0]["signature"] == ""
    assert msg["content"][1] == {"type": "text", "text": "结论"}


def test_nonstream_convert_tool_calls():
    data = {
        "choices": [{"message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "get_weather",
                             "arguments": '{"city": "北京"}'},
            }],
        }}],
    }
    msg = chat_completion_to_anthropic_message(data)
    assert msg["stop_reason"] == "tool_use"
    assert msg["content"] == [{
        "type": "tool_use", "id": "call_1",
        "name": "get_weather", "input": {"city": "北京"},
    }]


def test_nonstream_convert_tool_calls_with_null_content():
    """上游纯 tool_calls 响应带 content: null（OpenAI 系惯例）不应崩溃。"""
    data = {
        "choices": [{"message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "get_weather",
                             "arguments": '{"city": "北京"}'},
            }],
        }}],
    }
    msg = chat_completion_to_anthropic_message(data)
    assert msg["stop_reason"] == "tool_use"
    assert msg["content"] == [{
        "type": "tool_use", "id": "call_1",
        "name": "get_weather", "input": {"city": "北京"},
    }]
    assert not any(b.get("type") == "text" for b in msg["content"])


def test_think_split_only_leading():
    """正文中段的 <think> 不拆（避免误伤举例文本）。"""
    from buddy_proxy.anthropic_adapter import _split_leading_think
    thinking, text = _split_leading_think("前文 <think>xx</think> 后文")
    assert thinking == ""
    assert text == "前文 <think>xx</think> 后文"


# ---------------------------------------------------------------------------
# 单元测试：AnthropicStreamConverter 的 reasoning_content → thinking 块
# ---------------------------------------------------------------------------
def test_stream_converter_thinking_blocks():
    conv = AnthropicStreamConverter("glm-5.2")
    events = []
    for chunk in (
        {"choices": [{"delta": {"reasoning_content": "思"}}]},
        {"choices": [{"delta": {"reasoning_content": "考"}}]},
        {"choices": [{"delta": {"content": "答案"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ):
        events.extend(conv.feed_chunk(chunk))
    events.extend(conv.finish())

    names = [n for n, _ in events]
    assert names[0] == "message_start"
    # thinking 块（index 0）先于 text 块（index 1）
    starts = [(n, d) for n, d in events if n == "content_block_start"]
    assert starts[0][1]["index"] == 0
    assert starts[0][1]["content_block"]["type"] == "thinking"
    assert starts[1][1]["index"] == 1
    assert starts[1][1]["content_block"]["type"] == "text"

    think_deltas = [d for n, d in events
                    if n == "content_block_delta"
                    and d["delta"]["type"] == "thinking_delta"]
    assert [d["delta"]["thinking"] for d in think_deltas] == ["思", "考"]

    text_deltas = [d for n, d in events
                   if n == "content_block_delta"
                   and d["delta"]["type"] == "text_delta"]
    assert [d["delta"]["text"] for d in text_deltas] == ["答案"]

    delta = [d for n, d in events if n == "message_delta"][0]
    assert delta["delta"]["stop_reason"] == "end_turn"
    assert names[-1] == "message_stop"


def test_stream_converter_usage_includes_input_tokens():
    """finish() 的 message_delta usage 应同时带 input_tokens 与 output_tokens。"""
    conv = AnthropicStreamConverter("glm-5.2")
    events = []
    for chunk in (
        {"choices": [{"delta": {"content": "答案"}}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 15, "completion_tokens": 8, "total_tokens": 23}},
    ):
        events.extend(conv.feed_chunk(chunk))
    events.extend(conv.finish())

    delta = [d for n, d in events if n == "message_delta"][0]
    assert delta["usage"]["input_tokens"] == 15
    assert delta["usage"]["output_tokens"] == 8


# ---------------------------------------------------------------------------
# 集成测试：/v1/messages → trae provider
# ---------------------------------------------------------------------------
def _anthropic_body(stream: bool, tools: bool = False) -> dict:
    body = {
        "model": "trae/glm-5.2",
        "max_tokens": 1024,
        "system": "You are a helpful assistant.",
        "messages": [{"role": "user", "content": "你好"}],
        "stream": stream,
    }
    if tools:
        body["tools"] = [{
            "name": "get_weather",
            "description": "查询天气",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }]
    return body


def test_messages_nonstream_via_trae(client, trae_env):
    state, calls = trae_env
    r = client.post("/v1/messages", json=_anthropic_body(stream=False))
    assert r.status_code == 200
    msg = r.json()

    assert msg["type"] == "message"
    assert msg["role"] == "assistant"
    assert msg["model"] == "trae/glm-5.2"
    assert msg["content"][0]["type"] == "thinking"
    assert msg["content"][0]["thinking"] == "让我想想"
    assert msg["content"][1] == {"type": "text", "text": "你好！"}
    assert msg["stop_reason"] == "end_turn"
    assert msg["usage"]["output_tokens"] > 0

    # 上游收到的是转成 chat 格式的请求（system 置顶 + user 消息）
    assert calls, "send_trae_chat 应被调用"
    upstream_msgs = calls[0]["messages"]
    assert upstream_msgs[0]["role"] == "system"
    assert any(
        p.get("text") == "你好"
        for m in upstream_msgs if m["role"] == "user"
        for p in m["content"]
    )


def test_messages_stream_via_trae(client, trae_env):
    r = client.post("/v1/messages", json=_anthropic_body(stream=True))
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse_events(r.text)
    names = [n for n, _ in events]
    assert names[0] == "message_start"
    assert names[-1] == "" and events[-1][1] == "[DONE]"
    assert "message_stop" in names

    # thinking 块 + text 块都有，且 thinking 在前
    blocks = [d for n, d in events if n == "content_block_start"]
    assert [b["content_block"]["type"] for b in blocks] == ["thinking", "text"]

    think_text = "".join(
        d["delta"]["thinking"] for n, d in events
        if n == "content_block_delta" and d["delta"]["type"] == "thinking_delta"
    )
    assert think_text == "让我想想"

    body_text = "".join(
        d["delta"]["text"] for n, d in events
        if n == "content_block_delta" and d["delta"]["type"] == "text_delta"
    )
    assert body_text == "你好！"

    # stop_reason 与 usage（include_usage 对 anthropic 强制开启）
    delta = [d for n, d in events if n == "message_delta"][0]
    assert delta["delta"]["stop_reason"] == "end_turn"
    assert delta["usage"]["output_tokens"] == 20


def test_messages_tool_use_stream_via_trae(client, trae_env):
    state, _ = trae_env
    state._tool_mode = True

    r = client.post("/v1/messages", json=_anthropic_body(stream=True, tools=True))
    assert r.status_code == 200
    events = _parse_sse_events(r.text)
    names = [n for n, _ in events]

    # tool_use 块 + input_json_delta
    tool_start = [d for n, d in events
                  if n == "content_block_start"
                  and d["content_block"]["type"] == "tool_use"][0]
    assert tool_start["content_block"]["name"] == "get_weather"
    assert tool_start["content_block"]["id"]

    args = "".join(
        d["delta"]["partial_json"] for n, d in events
        if n == "content_block_delta" and d["delta"]["type"] == "input_json_delta"
    )
    assert json.loads(args) == {"city": "北京"}

    delta = [d for n, d in events if n == "message_delta"][0]
    assert delta["delta"]["stop_reason"] == "tool_use"
    assert "message_stop" in names


def test_messages_tool_use_nonstream_via_trae(client, trae_env):
    state, _ = trae_env
    state._tool_mode = True

    r = client.post("/v1/messages", json=_anthropic_body(stream=False, tools=True))
    assert r.status_code == 200
    msg = r.json()

    assert msg["stop_reason"] == "tool_use"
    tool_blocks = [b for b in msg["content"] if b["type"] == "tool_use"]
    assert len(tool_blocks) == 1
    assert tool_blocks[0]["name"] == "get_weather"
    assert tool_blocks[0]["input"] == {"city": "北京"}
    # 只有工具调用时不注入空响应兜底文案
    assert not any(
        b.get("type") == "text" and "空响应" in b.get("text", "")
        for b in msg["content"]
    )


def test_messages_tool_result_roundtrip(client, trae_env):
    """Claude Code 多轮循环：assistant tool_use + user tool_result 回传。"""
    state, calls = trae_env
    body = _anthropic_body(stream=False)
    body["messages"] = [
        {"role": "user", "content": "北京天气如何"},
        {
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "id": "toolu_1",
                "name": "get_weather",
                "input": {"city": "北京"},
            }],
        },
        {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "content": "晴，25 度",
            }],
        },
    ]
    r = client.post("/v1/messages", json=body)
    assert r.status_code == 200

    # 上游收到的 chat 消息：assistant 的 tool_use 序列化进正文、
    # tool_result 转成 [tool_result] 前缀的 user 消息
    upstream = calls[0]["messages"]
    roles = [m["role"] for m in upstream]
    assert "system" in roles
    assistant_texts = "".join(
        p.get("text", "")
        for m in upstream if m["role"] == "assistant"
        for p in m["content"]
    )
    assert "get_weather" in assistant_texts  # tool_use 被序列化保留
    result_texts = "".join(
        p.get("text", "")
        for m in upstream if m["role"] == "user"
        for p in m["content"]
    )
    assert "[tool_result" in result_texts
    assert "晴，25 度" in result_texts


def test_messages_stream_upstream_error(client, trae_env):
    """上游 event: error（如 4031 配额耗尽）→ anthropic event: error，不伪装成正文。"""
    state, _ = trae_env
    state._upstream_sse = (
        'event: error\ndata: {"code": 4031, "message": ""}\n\n'
        "event: done\ndata: {}\n\n"
    )
    r = client.post("/v1/messages", json=_anthropic_body(stream=True))
    assert r.status_code == 200

    events = _parse_sse_events(r.text)
    # 第一个事件是 error，不含 message_start / 伪正文（否则 Claude Code 会循环）
    names = [n for n, _ in events]
    assert names[0] == "error"
    assert "message_start" not in names
    err = events[0][1]
    assert err["type"] == "error"
    assert "4031" in err["error"]["message"]


def test_messages_nonstream_upstream_error(client, trae_env):
    """非流式上游错误 → Anthropic 标准错误形状（而非 FastAPI detail 包裹）。"""
    state, _ = trae_env
    state._upstream_sse = (
        'event: error\ndata: {"code": 4031, "message": ""}\n\n'
        "event: done\ndata: {}\n\n"
    )
    r = client.post("/v1/messages", json=_anthropic_body(stream=False))
    assert r.status_code == 502
    body = r.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "api_error"
    assert "4031" in body["error"]["message"]
