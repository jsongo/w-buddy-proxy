"""Trae 原生 function calling 通道测试（完全离线，不访问 Trae 上游）。

覆盖 buddy_proxy.trae.native_tools 的完整链路：
- 非流式 / 流式：原生 tool_calls SSE → OpenAI chat.completion（含真实 usage）
- 多轮结构化历史回放：assistant.tool_calls / role:tool 原样上报（不改写为文本）
- 4001 参数拒绝 → 自动回落文本协议（send_trae_chat），换号……换通道不影响结果
- 非 4001 上游错误 → 不回落，原样上报
- 工具载荷形状：parameters 序列化为 JSON 字符串（上游 Go schema 要求）

运行：
    PYTHONPATH=src python3 -m pytest test_trae_native_tools.py -v
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
from buddy_proxy.trae import provider as tp_impl
from buddy_proxy.trae.provider import TraeProvider


# ---------------------------------------------------------------------------
# 上游 SSE fixture：原生通道响应形态（event:output 携带结构化 tool_calls）
# ---------------------------------------------------------------------------
def _native_sse(outputs: list[dict], usage: bool = True) -> str:
    parts = []
    for o in outputs:
        parts.append("event: output")
        parts.append("data: " + json.dumps(o, ensure_ascii=False))
        parts.append("")
    if usage:
        parts += [
            "event: token_usage",
            'data: {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}',
            "",
        ]
    parts += ["event: done", "data: {}", "", ""]
    return "\n".join(parts)


SSE_NATIVE_TOOL_CALL = _native_sse([
    {"reasoning_content": "我来查一下天气"},
    {"response": "", "tool_calls": [
        {"index": 0, "id": "call_native_01", "type": "function",
         "function_call": {"name": "get_weather", "arguments": "{\"city\": \"北京\"}",
                           "partial_arguments": None, "namespace": None}},
    ]},
])

SSE_NATIVE_REJECTED = (
    'event: error\ndata: {"code": 4001, "message": "param is invalid"}\n\n'
    "event: done\ndata: {}\n\n"
)

SSE_NATIVE_ERROR_4031 = (
    'event: error\ndata: {"code": 4031, "message": ""}\n\n'
    "event: done\ndata: {}\n\n"
)

SSE_LEGACY_TOOL_CALL = (
    'event: output\ndata: {"response": "<tool_call>\\n{\\"name\\": \\"get_weather\\", '
    '\\"arguments\\": {\\"city\\": \\"北京\\"}}\\n</tool_call>"}\n\n'
    "event: done\ndata: {}\n\n"
)


def _chat_body(stream: bool = False, tools: bool = True, multi_turn: bool = False) -> dict:
    messages = [
        {"role": "user", "content": "北京天气如何？"},
    ]
    if multi_turn:
        messages = [
            {"role": "user", "content": "北京天气如何？"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_hist_01", "type": "function",
                 "function": {"name": "get_weather", "arguments": "{\"city\": \"北京\"}"}}]},
            {"role": "tool", "tool_call_id": "call_hist_01", "name": "get_weather",
             "content": "{\"temp_c\": 25, \"condition\": \"sunny\"}"},
            {"role": "user", "content": "总结一下"},
        ]
    body: dict = {"model": "deepseek-v4-flash", "stream": stream, "messages": messages}
    if stream:
        body["stream_options"] = {"include_usage": True}
    if tools:
        body["tools"] = [{
            "type": "function",
            "function": {"name": "get_weather", "description": "查询城市天气",
                         "parameters": {"type": "object",
                                        "properties": {"city": {"type": "string"}},
                                        "required": ["city"]}}},
        ]
    return body


# ---------------------------------------------------------------------------
# fixture：假 ProxyState + mock 原生/legacy 上游
# ---------------------------------------------------------------------------
@pytest.fixture()
def native_env(monkeypatch):
    provider = TraeProvider()
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

    native_calls: list[dict] = []
    legacy_calls: list[dict] = []

    def fake_native(native_msgs, model, stream, tools):
        native_calls.append({"messages": native_msgs, "model": model,
                             "stream": stream, "tools": tools})
        if getattr(state, "_native_sse", None) is not None:
            return state._native_sse
        return SSE_NATIVE_TOOL_CALL

    def fake_legacy(messages, model, stream, base_url=None):
        legacy_calls.append({"messages": messages, "model": model, "stream": stream})
        return getattr(state, "_legacy_sse", None) or SSE_LEGACY_TOOL_CALL

    monkeypatch.setattr(tp_impl, "_send_native_chat", fake_native)
    monkeypatch.setattr(tp_impl, "send_trae_chat", fake_legacy)
    monkeypatch.setattr(tp_impl, "_auth", lambda: ("fake-token", "fake-uid"))
    return state, native_calls, legacy_calls


@pytest.fixture()
def client(native_env):
    return TestClient(m.app)


def _parse_sse_events(text: str) -> list[tuple[str, dict | str]]:
    events, cur_event, cur_data = [], "", []

    def flush():
        if cur_data:
            data_str = "\n".join(cur_data)
            events.append((cur_event, "[DONE]" if data_str == "[DONE]" else json.loads(data_str)))
            cur_data.clear()

    for line in text.splitlines():
        line = line.strip()
        if not line:
            flush(); cur_event = ""
        elif line.startswith("event:"):
            cur_event = line[6:].strip()
        elif line.startswith("data:"):
            cur_data.append(line[5:].strip())
    flush()
    return events


# ---------------------------------------------------------------------------
# 非流式 / 流式
# ---------------------------------------------------------------------------
def test_native_nonstream_tool_calls(client, native_env):
    state, native_calls, _ = native_env
    r = client.post("/v1/chat/completions", json=_chat_body(stream=False))
    assert r.status_code == 200
    resp = r.json()
    ch = resp["choices"][0]
    assert ch["finish_reason"] == "tool_calls"
    tc = ch["message"]["tool_calls"]
    assert len(tc) == 1
    assert tc[0]["id"] == "call_native_01"
    assert tc[0]["function"]["name"] == "get_weather"
    assert json.loads(tc[0]["function"]["arguments"]) == {"city": "北京"}
    # 原生通道带真实 usage
    assert resp["usage"]["prompt_tokens"] == 10
    assert len(native_calls) == 1


def test_native_stream_tool_calls(client, native_env):
    state, native_calls, _ = native_env
    r = client.post("/v1/chat/completions", json=_chat_body(stream=True))
    assert r.status_code == 200
    events = _parse_sse_events(r.text)
    assert events[-1] == ("", "[DONE]")
    tool_deltas = []
    finish = None
    usage = None
    for ev, data in events:
        if not isinstance(data, dict):
            continue
        if "usage" in data:
            usage = data["usage"]
        for ch in data.get("choices", []):
            if (ch.get("delta") or {}).get("tool_calls"):
                tool_deltas.extend(ch["delta"]["tool_calls"])
            if ch.get("finish_reason"):
                finish = ch["finish_reason"]
    assert finish == "tool_calls"
    assert len(tool_deltas) == 1
    assert tool_deltas[0]["index"] == 0
    assert tool_deltas[0]["function"]["name"] == "get_weather"
    assert usage is not None and usage["total_tokens"] == 30


# ---------------------------------------------------------------------------
# 多轮结构化历史回放：不改写为文本协议
# ---------------------------------------------------------------------------
def test_native_history_replay_structured(client, native_env):
    state, native_calls, _ = native_env
    r = client.post("/v1/chat/completions", json=_chat_body(stream=False, multi_turn=True))
    assert r.status_code == 200
    sent = native_calls[0]["messages"]
    # assistant.tool_calls 保持结构化 function_call 形状
    asst = [m for m in sent if m.get("role") == "assistant"]
    assert asst and asst[0]["tool_calls"][0]["function_call"]["name"] == "get_weather"
    assert asst[0]["tool_calls"][0]["id"] == "call_hist_01"
    # role:tool 保持 tool 角色 + tool_call_id，不转 user + [tool_result] 前缀
    tool = [m for m in sent if m.get("role") == "tool"]
    assert tool and tool[0]["tool_call_id"] == "call_hist_01"
    assert tool[0]["content"][0]["text"] == "{\"temp_c\": 25, \"condition\": \"sunny\"}"
    # 不注入工具教学/压制指令
    sys_msgs = [m for m in sent if m.get("role") == "system"]
    assert not sys_msgs or "tool_call" not in json.dumps(sys_msgs, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 工具载荷形状：parameters 序列化为 JSON 字符串
# ---------------------------------------------------------------------------
def test_native_tools_payload_string_params(client, native_env):
    """参数 stringify 发生在 _build_native_body 内部，直接单测该转换。"""
    from buddy_proxy.trae.native_tools import _native_tools_payload

    state, native_calls, _ = native_env
    client.post("/v1/chat/completions", json=_chat_body(stream=False))
    # 请求链路上 tools 原样传入 native 发送器
    assert len(native_calls[0]["tools"]) == 1
    payload = _native_tools_payload(_chat_body()["tools"])
    fn = payload[0]["function"]
    assert fn["name"] == "get_weather"
    assert isinstance(fn["parameters"], str)
    assert json.loads(fn["parameters"])["required"] == ["city"]


# ---------------------------------------------------------------------------
# 4001 → 自动回落文本协议
# ---------------------------------------------------------------------------
def test_native_4001_falls_back_to_legacy(client, native_env):
    state, native_calls, legacy_calls = native_env
    state._native_sse = SSE_NATIVE_REJECTED
    r = client.post("/v1/chat/completions", json=_chat_body(stream=False))
    assert r.status_code == 200
    resp = r.json()
    ch = resp["choices"][0]
    # 文本协议路径解析出同样的调用
    assert ch["finish_reason"] == "tool_calls"
    assert ch["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert len(native_calls) == 1 and len(legacy_calls) == 1


def test_native_4001_stream_falls_back(client, native_env):
    state, native_calls, legacy_calls = native_env
    state._native_sse = SSE_NATIVE_REJECTED
    r = client.post("/v1/chat/completions", json=_chat_body(stream=True))
    assert r.status_code == 200
    events = _parse_sse_events(r.text)
    finish = [d.get("choices", [{}])[0].get("finish_reason")
              for ev, d in events if isinstance(d, dict) and d.get("choices")]
    assert "tool_calls" in finish
    assert len(native_calls) == 1 and len(legacy_calls) == 1


# ---------------------------------------------------------------------------
# 非 4001 上游错误 → 不回落，原样上报
# ---------------------------------------------------------------------------
def test_native_non_4001_error_no_fallback(client, native_env):
    state, native_calls, legacy_calls = native_env
    state._native_sse = SSE_NATIVE_ERROR_4031
    r = client.post("/v1/chat/completions", json=_chat_body(stream=False))
    assert r.status_code == 502
    assert "4031" in r.json().get("detail", "") or "4031" in r.text
    assert legacy_calls == []
