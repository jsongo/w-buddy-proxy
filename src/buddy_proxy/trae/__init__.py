"""Trae 子包：buddy_proxy 的 Trae 公有云代理实现。

模块划分（依赖方向自上而下，禁止反向）：

- ``config``         常量、版本头、模型映射、环境开关、调试/心跳
- ``auth_storage``   Trae IDE 本地存储解密（tc 加密格式）
- ``benefits_api``   签到 / 积分 / 权益用量上游 API
- ``credentials``    凭证加载（Work 优先，IDE 解密回退）与两种请求头
- ``leak_guard``     agent 预设泄漏防护（压制指令 + 正文/流式清洗）
- ``text_toolcall``  文本协议工具调用解析（JSON 修复、流式分片、泄漏闸门）
- ``text_protocol``  文本协议请求侧转换（guard/教学注入、messages 改写）
- ``native_tools``   原生 function calling 通道（chat_v3 直通，2026-09 实测）
- ``transport``      legacy 文本协议 HTTP 发送（端点回退 / Work 重试）
- ``sse``            SSE 解析与 Anthropic 流包装
- ``provider``       TraeProvider 编排入口
- ``cli``            账号工具命令行
"""

from __future__ import annotations

from .auth_storage import decrypt_auth_data, decrypt_storage_value, find_auth_data
from .benefits_api import (
    checkin_device_id,
    claim_checkin_credits,
    fetch_checkin_status,
    fetch_ent_usage,
)
from .cli import _cli
from .config import (
    BASE_URL_CN,
    BASE_URL_SG,
    ENDPOINTS,
    MODEL_MAP,
    MODEL_TIERS,
    TRAE_HEARTBEAT_INTERVAL,
    X_APP_ID,
    _map_model,
)
from .credentials import _auth, _build_headers, _load_work_cred, _work_headers
from .leak_guard import _AGENT_GUARD, _StreamLeakCleaner, _sanitize_agent_leak
from .native_tools import (
    _NativeToolAccumulator,
    _native_messages,
    _native_rejected,
    _native_tools_payload,
    _send_native_chat,
)
from .provider import TraeProvider
from .sse import _parse_sse, _trae_error_text, _wrap_anthropic_stream
from .text_protocol import (
    _build_chat_body,
    _build_tools_system,
    _extract_prompt,
    _looks_like_agent_request,
    _serialize_tool_calls,
)
from .text_toolcall import (
    _TC_CLOSE,
    _TC_OPEN,
    _StreamToolCallSplitter,
    _parse_tool_calls,
    _tool_names,
)
from .transport import _WORK_CHAT_MAX_ATTEMPTS, _send_trae_work_chat, send_trae_chat

__all__ = [
    "TraeProvider",
    "BASE_URL_CN",
    "BASE_URL_SG",
    "ENDPOINTS",
    "MODEL_MAP",
    "MODEL_TIERS",
    "TRAE_HEARTBEAT_INTERVAL",
    "X_APP_ID",
    "find_auth_data",
    "decrypt_auth_data",
    "decrypt_storage_value",
    "checkin_device_id",
    "fetch_checkin_status",
    "claim_checkin_credits",
    "fetch_ent_usage",
    "send_trae_chat",
]
