"""FastAPI 路由与请求处理。

所有 ``@app`` 装饰的路由统一放这里，通过 ``from .state import app`` 引用共享 app 实例。
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

from buddy_proxy.state import (
    app,
    _get_state_or_none,
    diagnostic,
    get_state,
)
from buddy_proxy.model_list import load_models_from_local_config, model_to_codex_format
from buddy_proxy.codebuddy_provider import (
    HAS_PROJECTION,
    anthropic_to_chat,
    body_summary,
    forward_chat,
    log_client_request,
    project_responses_chat_body,
    responses_request_to_chat,
)

# 尝试导入高级功能模块（可选）
try:
    from buddy_proxy.desensitize import desensitize_body
    HAS_DESENSITIZE = True
except ImportError:
    HAS_DESENSITIZE = False

    def desensitize_body(body, **kwargs):
        return body


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """全局兜底异常日志：任何未捕获异常都记录（便于定位问题），再返回 500。"""
    try:
        state = _get_state_or_none()
        if state is not None:
            import traceback
            tb = traceback.format_exc()
            if state.logger:
                state.logger.error(
                    "unhandled_exception: %s %s -> %s\n%s",
                    request.method, request.url.path, exc, tb,
                )
            if state.json_logger:
                state.write_log(
                    "unhandled_exception",
                    method=request.method,
                    path=request.url.path,
                    error=str(exc),
                )
    except Exception:
        pass  # 日志失败不影响响应
    # 不回显 exc 原文（避免内部细节/路径信息暴露给客户端），详情在服务端日志里
    return JSONResponse(
        status_code=500,
        content={"error": {"message": "internal error", "type": "internal_error"}},
    )


@app.get("/health")
async def health():
    state = get_state()
    auth = {} if state.mock_dir is not None else (state.client.session.get("auth") or {})
    expires = int(auth.get("expiresAt") or 0)
    # 附加各 provider 的健康信息（兼容无 providers 属性的旧构造）
    providers_health = {pid: p.health() for pid, p in getattr(state, "providers", {}).items()}
    return {
        "status": "ok",
        "authenticated": bool(auth.get("accessToken")),
        "token_valid": not expires or expires > int(time.time() * 1000),
        "uptime_seconds": int(time.time() - state.started_at),
        "providers": providers_health,
    }


@app.get("/v1/models")
async def list_models():
    state = get_state()
    # 从本地配置文件加载模型列表（离线可靠，无需认证）
    data = load_models_from_local_config()
    # 标记通道归属：静态表中的模型走默认 CodeBuddy 通道
    for m in data:
        m.setdefault("provider", "codebuddy")

    # 合并其它 provider 的模型（如豆包/Trae）。
    # provider 字段标识该模型由哪个上游通道提供（codebuddy/trae/doubao），
    # 与 owned_by（上游厂牌，如 zhipu）区分，便于客户端辨识。
    for provider in getattr(state, "providers", {}).values():
        for m in provider.models():
            data.append({
                "id": m.get("id"),
                "name": m.get("description") or m.get("id"),
                "vendor": provider.id,
                "owned_by": provider.id,
                "provider": provider.id,
            })

    # 记录模型列表请求
    diagnostic(
        "models_list_request",
        models_count=len(data),
        source="local_config"
    )

    # 标准 OpenAI 格式：/v1/models 的 data 数组（客户端按此解析）
    openai_models = [
        {
            "id": m.get("id", "unknown"),
            "object": "model",
            "created": 1720872952,
            "owned_by": m.get("vendor") or "codebuddy",
            # 通道归属：codebuddy（默认）/ trae / doubao ...
            "provider": m.get("provider", "codebuddy"),
            "name": m.get("name") or m.get("id", "unknown"),
            "display_name": m.get("name") or m.get("id", "unknown"),
            "credits": m.get("credits"),
            "tags": m.get("tags", []),
        }
        for m in data
    ]

    # 扩展：Codex 兼容的完整模型元数据（含上下文窗口、能力等）
    codex_models = [model_to_codex_format(m) for m in data]

    return {"object": "list", "data": openai_models, "models": codex_models}


async def parse_request_body(request: Request) -> Any:
    """解析 JSON 请求体，兼容 GBK/cp936/latin-1 等非 UTF-8 编码。

    某些客户端（如 Pi）偶尔以 GBK 编码发送请求体，而 Starlette 的
    ``request.json()`` 内部是 ``json.loads(raw_bytes)``，默认按 UTF-8 解码，
    遇非 UTF-8 字节会直接抛 ``UnicodeDecodeError``（``ValueError`` 子类，
    非 ``JSONDecodeError``），导致 500。此处改为按
    utf-8 → gbk → cp936 → latin-1 依次解码再解析，彻底失败时返回 400。
    """
    raw = await request.body()
    for encoding in ("utf-8", "gbk", "cp936", "latin-1"):
        try:
            text = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            continue
    # latin-1 能解码任意字节，故 decode 不会失败；走到这里仅当 JSON 结构非法。
    from fastapi import HTTPException
    raise HTTPException(
        status_code=400,
        detail={
            "error": {
                "message": "请求体不是有效的 JSON（已尝试 utf-8/gbk/cp936/latin-1 解码）",
                "type": "invalid_request_body",
            }
        },
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    state = get_state()
    body = await parse_request_body(request)

    log_client_request("POST", "/v1/chat/completions", body)
    diagnostic("request", protocol="openai", **body_summary(body))

    # 调试抓包：WB_DEBUG_DUMP=1 时落完整请求体（含 messages 全文、tools），
    # 用于排查下游 agent 的协议细节（默认关闭，不影响生产日志）
    if os.environ.get("WB_DEBUG_DUMP"):
        state.write_log("debug_full_request", body=body)

    return await forward_chat(body, "openai")


@app.post("/v1/responses")
async def create_response(request: Request):
    state = get_state()
    body = await parse_request_body(request)

    log_client_request("POST", "/v1/responses", body)

    # 转换 Responses → Chat
    chat_body = responses_request_to_chat(body)

    # 消息压缩优化（如果启用）
    if state.enable_optimize_context and HAS_PROJECTION:
        chat_body, proj_stats = project_responses_chat_body(chat_body)
        diagnostic("projection_applied", protocol="responses", **proj_stats)

    # 过滤无效的工具定义
    tools = chat_body.get("tools", [])
    if tools:
        original_count = len(tools)
        filtered_tools = []
        filtered_names = []

        for tool in tools:
            # 1. 过滤非 function 类型
            if tool.get("type") != "function":
                filtered_names.append(f"{tool.get('type', 'unknown')} (非function类型)")
                continue

            # 2. 过滤空 parameters
            func = tool.get("function", {})
            params = func.get("parameters", {})
            if not params or not isinstance(params, dict) or len(params) == 0:
                filtered_names.append(f"{func.get('name', 'unknown')} (空parameters)")
                continue

            # 3. 检查 parameters 是否有 type 字段
            if "type" not in params:
                filtered_names.append(f"{func.get('name', 'unknown')} (缺少type)")
                continue

            filtered_tools.append(tool)

        chat_body["tools"] = filtered_tools

        if filtered_names:
            diagnostic("tools_filtered",
                      original=original_count,
                      kept=len(filtered_tools),
                      filtered=filtered_names)

    diagnostic("request", protocol="responses", **body_summary(chat_body))
    return await forward_chat(chat_body, "responses", original=body)


@app.post("/v1/messages")
async def create_message(request: Request):
    state = get_state()
    body = await parse_request_body(request)

    log_client_request("POST", "/v1/messages", body)

    # 转换 Anthropic → Chat
    chat_body = anthropic_to_chat(body)
    diagnostic("request", protocol="anthropic", **body_summary(chat_body))

    return await forward_chat(chat_body, "anthropic", original=body)
