"""关键接口冒烟测试（完全离线，不访问远程 CodeBuddy 后端）。

覆盖：/health、/v1/models、/v1/chat/completions、/v1/responses、/v1/messages。
所有依赖远程 / 全局状态的调用都被 monkeypatch 掉，保证测试可离线、可重复运行。

运行：
    .venv/bin/python -m pytest test_endpoints_smoke.py -v
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
from buddy_proxy import codebuddy_provider as cbp


# ---------------------------------------------------------------------------
# fixture：构造一个假的 ProxyState，并替换模块级 get_state()
# （拆分后 get_state/collect_upstream/stream_upstream 分别位于 state 与
#  codebuddy_provider 模块，monkeypatch 目标随之调整。）
# ---------------------------------------------------------------------------
def _make_state(client_mock=None):
    """构造一个最小可用的假 ProxyState（SimpleNamespace）。"""
    client = client_mock
    if client is None:
        client = mock.MagicMock()
        client.endpoint = "https://fake.endpoint.invalid"
        client.auth_headers.return_value = {"X-IDE-Fake": "1"}
        client.session = {"auth": {"accessToken": "test-token", "expiresAt": int((time.time() + 3600) * 1000)}}

    return SimpleNamespace(
        client=client,
        providers={},
        mock_dir=None,
        started_at=time.time(),
        enable_desensitize=False,
        enable_optimize_context=False,
        verbose_llm=False,
        write_log=mock.MagicMock(),
        ensure_auth=mock.MagicMock(),
        logger=mock.MagicMock(),
        json_logger=mock.MagicMock(),
        runtime_info={"app_version": "test", "system_version": "test", "python_version": "test", "machine": "test"},
    )


@pytest.fixture()
def proxy_state(monkeypatch):
    """为测试提供替换 get_state 的假 state。

    拆分后 get_state() 在 state 模块内读取模块级 proxy_state 变量，
    故直接替换 st.proxy_state 即可让所有模块（routes/codebuddy_provider
    等）拿到假 state。
    """
    state = _make_state()
    monkeypatch.setattr(st, "proxy_state", state)
    return state


@pytest.fixture()
def client(proxy_state):
    """FastAPI TestClient，指向真实的 app。"""
    return TestClient(m.app)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["authenticated"] is True
    assert body["token_valid"] is True
    assert isinstance(body["uptime_seconds"], int)


def test_health_unauthenticated(monkeypatch):
    """无 token 时 authenticated 应为 False。"""
    state = _make_state()
    state.client.session = {}  # 无 auth
    monkeypatch.setattr(st, "proxy_state", state)
    c = TestClient(m.app)
    body = c.get("/health").json()
    assert body["authenticated"] is False
    # expires 为 0 时按「不过期」处理，故 token_valid 为 True
    assert body["token_valid"] is True


# ---------------------------------------------------------------------------
# /v1/models（真实静态加载，不 mock 数据）
# ---------------------------------------------------------------------------
def test_models_shape_and_count(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert len(body["data"]) >= 10
    assert len(body["models"]) == len(body["data"])


def test_models_has_required_ids(client):
    """用户要求的模型 id 必须都在静态配置里。"""
    required = {
        "glm-5.3", "glm-5.3-flash", "hy3", "hy4-preview",
        "minimax-m3", "kimi-k3", "kimi-k2.7",
        "deepseek-v4-flash", "deepseek-v4-pro",
    }
    body = client.get("/v1/models").json()
    ids = {x["id"] for x in body["data"]}
    assert required <= ids, f"缺少模型: {required - ids}"


def test_models_credits_and_name(client):
    """data 和 models 都必须透出 name 与 credits，且与 config 一致。"""
    body = client.get("/v1/models").json()
    by_id = {x["id"]: x for x in body["data"]}

    checks = {
        "glm-5.3": ("x0.79 credits", "GLM-5.3"),
        "glm-5.3-flash": ("x0.06 credits", "GLM-5.3-Flash"),
        "hy4-preview": ("x0.29 credits", "Hy4 preview"),
        "minimax-m3": ("x0.25 credits", "MiniMax-M3"),
        "deepseek-v4-flash": ("x0.17 credits", "Deepseek-V4-Flash"),
        "deepseek-v4-pro": ("x0.51 credits", "Deepseek-V4-Pro"),
        "kimi-k2.7": ("x0.57 credits", "Kimi-K2.7-Code"),
    }
    for mid, (credit, name) in checks.items():
        entry = by_id.get(mid)
        assert entry is not None, f"{mid} 缺失"
        assert entry["name"] == name, f"{mid} name={entry.get('name')!r}"
        assert entry["credits"] == credit, f"{mid} credits={entry.get('credits')!r}"


def test_models_codex_metadata(client):
    """codex models 要带 context_window / max_context_window / credits。"""
    body = client.get("/v1/models").json()
    sample = next(x for x in body["models"] if x["id"] == "deepseek-v4-pro")
    assert sample.get("max_context_window") is not None
    assert sample.get("context_window") == sample.get("max_context_window")
    assert sample["credits"] == "x0.51 credits"


# ---------------------------------------------------------------------------
# /v1/chat/completions（非流式，monkeypatch collect_upstream）
# ---------------------------------------------------------------------------
def _fake_collected():
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1720872952,
        "model": "glm-5.3",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "你好，我是测试回复"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def test_chat_completions_nonstream(client, monkeypatch):
    async def fake_collect(*args, **kwargs):
        return _fake_collected()
    monkeypatch.setattr(cbp, "collect_upstream", fake_collect)

    r = client.post("/v1/chat/completions", json={
        "model": "glm-5.3",
        "messages": [{"role": "user", "content": "hello"}],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "你好，我是测试回复"
    assert body["model"] == "glm-5.3"


# ---------------------------------------------------------------------------
# /v1/responses（注意：该端点返回 SSE 事件流，即使非流式请求也如此）
# ---------------------------------------------------------------------------
def test_responses_nonstream(client, monkeypatch):
    """/v1/responses 走流式转发，返回 SSE 事件流。patch stream_upstream 模拟上游。"""
    async def fake_stream(url, headers, body, protocol, original):
        # 模拟上游已完成的响应，产生 responses 协议的 response.completed 事件
        yield (
            "event: response.completed\ndata: "
            + json.dumps({"type": "response.completed", "response": {"id": "resp_test", "status": "completed"}}, ensure_ascii=False)
            + "\n\n"
        ).encode()
        yield b"data: [DONE]\n\n"
    monkeypatch.setattr(cbp, "stream_upstream", fake_stream)

    r = client.post("/v1/responses", json={
        "model": "glm-5.3",
        "input": "hello",
    })
    assert r.status_code == 200
    assert "text/event-stream" in r.headers.get("content-type", "")
    text = r.text
    # SSE 事件流被透传给客户端
    assert "response.completed" in text
    assert "[DONE]" in text


# ---------------------------------------------------------------------------
# /v1/messages（Anthropic /v1/messages）
# ---------------------------------------------------------------------------
def test_messages_nonstream(client, monkeypatch):
    async def fake_collect(*args, **kwargs):
        return _fake_collected()
    monkeypatch.setattr(cbp, "collect_upstream", fake_collect)

    r = client.post("/v1/messages", json={
        "model": "claude-4.0-sonnet",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 256,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["content"]  # 至少一个 content block


# ---------------------------------------------------------------------------
# 请求体非 UTF-8（GBK）也能解析
# ---------------------------------------------------------------------------
def test_gbk_request_body(client, monkeypatch):
    """parse_request_body 要能处理 GBK 编码的请求体，不应 500。"""
    async def fake_collect(*args, **kwargs):
        return _fake_collected()
    monkeypatch.setattr(cbp, "collect_upstream", fake_collect)

    gbk_payload = (
        '{"model":"glm-5.3","messages":[{"role":"user","content":"\u4f60\u597d"}]}'
    ).encode("gbk")

    r = client.post(
        "/v1/chat/completions",
        content=gbk_payload,
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 200
