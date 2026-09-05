"""Trae Provider —— 兼容 shim，实现已拆分到 :mod:`buddy_proxy.trae` 子包。

历史入口保留：本模块曾承载 Trae 代理全部实现（认证/调用/工具协议转换），
现按职责拆到 ``buddy_proxy/trae/`` 下 12 个模块（见该包 docstring 的模块图）。
这里平铺再导出全部既有名字，保证历史引用方零改动：

- ``from buddy_proxy.trae_provider import TraeProvider, _parse_tool_calls, ...``
- ``python -m buddy_proxy.trae_provider status|claim|usage|chat``

新代码请直接从 ``buddy_proxy.trae`` 或对应子模块导入。
"""

from __future__ import annotations

from .trae.auth_storage import (  # noqa: F401
    decrypt_auth_data,
    decrypt_storage_value,
    find_auth_data,
)
from .trae.benefits_api import (  # noqa: F401
    checkin_device_id,
    claim_checkin_credits,
    fetch_checkin_status,
    fetch_ent_usage,
)
from .trae.cli import _cli  # noqa: F401
from .trae.config import (  # noqa: F401
    BASE_URL_CN,
    BASE_URL_SG,
    ENDPOINTS,
    MODEL_MAP,
    MODEL_TIERS,
    TRAE_HEARTBEAT_INTERVAL,
    X_APP_ID,
    _debug_dump,
    _heartbeat_text,
    _map_model,
)
from .trae.credentials import (  # noqa: F401
    _auth,
    _build_headers,
    _load_work_cred,
    _work_headers,
)
from .trae.leak_guard import (  # noqa: F401
    _AGENT_GUARD,
    _StreamLeakCleaner,
    _sanitize_agent_leak,
)
from .trae.native_tools import (  # noqa: F401
    _NativeToolAccumulator,
    _native_messages,
    _native_rejected,
    _native_tools_payload,
    _send_native_chat,
)
from .trae.provider import TraeProvider  # noqa: F401
from .trae.sse import (  # noqa: F401
    _parse_sse,
    _trae_error_text,
    _wrap_anthropic_stream,
)
from .trae.text_protocol import (  # noqa: F401
    _build_chat_body,
    _build_tools_system,
    _extract_prompt,
    _looks_like_agent_request,
    _serialize_tool_calls,
)
from .trae.text_toolcall import (  # noqa: F401
    _TC_CLOSE,
    _TC_OPEN,
    _StreamToolCallSplitter,
    _parse_tool_calls,
    _tool_names,
)
from .trae.transport import (  # noqa: F401
    _WORK_CHAT_MAX_ATTEMPTS,
    _send_trae_work_chat,
    send_trae_chat,
)

_SUBMODULES = ("config", "auth_storage", "benefits_api", "credentials", "leak_guard",
               "text_toolcall", "text_protocol", "native_tools", "transport",
               "sse", "provider", "cli")


def __getattr__(name: str):
    """兜底再导出：显式清单之外的历史名字（私有常量/辅助函数等）从子包解析。

    保证旧代码对 trae_provider.<任意历史顶层名字> 的读取/patch 不断链。
    """
    import importlib

    for sub in _SUBMODULES:
        try:
            mod = importlib.import_module(f"{__package__}.trae.{sub}")
        except ImportError:
            continue
        if hasattr(mod, name):
            return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    _cli()
