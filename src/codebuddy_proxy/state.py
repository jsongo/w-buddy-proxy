"""全局状态管理：ProxyState、FastAPI app 实例、以及跨模块共享的全局变量。

把 ``app`` / ``proxy_state`` / ``remote_config_cache`` 三个模块级全局统一放在这里，
其它模块（routes、codebuddy_provider、model_list）通过 ``from .state import ...`` 引用，
避免拆分后出现循环导入。
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any, Optional

from fastapi import FastAPI, HTTPException

from codebuddy_proxy.codebuddy_client_demo import CodeBuddyClient
from codebuddy_proxy.providers import BaseProvider
from codebuddy_proxy.logging_setup import get_runtime_info


class ProxyState:
    """管理 proxy 的全局状态：认证、日志、配置。"""

    def __init__(
        self,
        client: CodeBuddyClient,
        mock_dir: Optional[Any],
        log_file: Optional[Any],
        enable_desensitize: bool = False,
        enable_optimize_context: bool = False,
        verbose_llm: bool = False,
        logger: Optional[logging.Logger] = None,
        json_logger: Optional[logging.Logger] = None,
        providers: Optional[dict[str, BaseProvider]] = None,
    ):
        self.client = client
        # 多 provider 支持：除默认 CodeBuddy 外的其它上游源（按 provider.id 索引）
        self.providers: dict[str, BaseProvider] = providers or {}
        self.mock_dir = mock_dir
        self.log_file = log_file
        self.enable_desensitize = enable_desensitize
        self.enable_optimize_context = enable_optimize_context
        self.verbose_llm = verbose_llm
        self.logger = logger
        self.json_logger = json_logger
        self.runtime_info = get_runtime_info()
        self.started_at = time.time()

    def ensure_auth(self) -> None:
        """确保已认证；认证失败（token 过期/网络错误/登录未完成）返回结构化 401 而非 500。"""
        if self.mock_dir is not None:
            return
        try:
            self.client.ensure_authenticated()
        except HTTPException:
            raise  # 已是结构化异常，原样透传（如 503 proxy not initialized）
        except Exception as exc:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "message": "认证失败：token 无效或已过期，请重新登录（--login）",
                        "type": "authentication_error",
                        "details": str(exc)[:200],
                    }
                },
            )

    def write_log(self, event: str, **kwargs) -> None:
        if self.json_logger is None:
            return
        try:
            record = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "event": event,
                **self.runtime_info,
                **kwargs,
            }
            self.json_logger.info(json.dumps(record, ensure_ascii=False))
        except Exception:
            pass

    def write_body_log(self, event: str, body: bytes, **kwargs) -> None:
        if self.json_logger is None:
            return
        try:
            text = body.decode("utf-8", errors="replace")
            record = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "event": event,
                **self.runtime_info,
                "body_bytes": len(body),
                "body_text": text,
                **kwargs,
            }
            self.json_logger.info(json.dumps(record, ensure_ascii=False))
        except Exception:
            pass


# ============================================================================
# FastAPI 应用（模块级单例，路由模块通过 from .state import app 引用）
# ============================================================================

app = FastAPI(title="CodeBuddy Proxy (FastAPI)", version="2.0")

# 全局状态（在 main() 中初始化）
proxy_state: Optional[ProxyState] = None

# 远程配置缓存实例（在 main() 中初始化；个人版用户不启用动态模型列表时保持 None）
remote_config_cache: Any = None


def _get_state_or_none() -> Optional[ProxyState]:
    """get_state 的容错版本：未初始化时返回 None（用于日志场景）。"""
    return proxy_state


def get_state() -> ProxyState:
    if proxy_state is None:
        raise HTTPException(
            status_code=503,
            detail={"error": {"message": "proxy not initialized", "type": "internal_error"}},
        )
    return proxy_state


def diagnostic(event: str, **kwargs) -> None:
    """输出诊断日志到 logger。"""
    state = get_state()
    if state.logger:
        state.logger.info(f"{event}: {json.dumps(kwargs, ensure_ascii=False)}")


# 安全词检测统一关键词（中英混合）
_SAFETY_KEYWORDS = ("sensitive", "cannot respond", "敏感内容", "无法响应", "unable to")


def is_policy_blocked(text: str) -> bool:
    """检测文本是否包含安全策略拦截标记"""
    return any(marker in text.lower() for marker in _SAFETY_KEYWORDS)


def text_summary(value: str) -> dict[str, Any]:
    return {
        "content_length": len(value),
        "content_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest()[:16],
        "safety_message_detected": is_policy_blocked(value),
    }
