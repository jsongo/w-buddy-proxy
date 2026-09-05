"""Trae HTTP 传输层：legacy 文本协议的端点回退与 Work 通道重试发送。"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any

from fastapi import HTTPException

from .config import (
    BASE_URL_CN,
    ENDPOINTS,
    _WORK_CHAT_MAX_ATTEMPTS,
    _WORK_FUNCTION_OVERRIDE,
    _debug_dump,
    _map_model,
)
from .credentials import _auth, _build_headers, _load_work_cred, _work_headers
from .text_protocol import _build_chat_body

log = logging.getLogger(__name__)

def send_trae_chat(
    messages: list[dict[str, Any]],
    model: str,
    stream: bool,
    base_url: str = BASE_URL_CN,
) -> str:
    """直连 Trae API，返回原始 SSE 文本。

    3 级端点回退。认证失败抛 HTTPException(401)，其余抛 RuntimeError。
    若使用 Work 凭证（~/.ethan/trae_work.json），自动走 Work 通道
    （function=solo_work_lite + api.trae.com.cn）。
    """
    work = _load_work_cred()
    if work and work.get("access_token"):
        return _send_trae_work_chat(messages, model, stream, work)

    token, user_id = _auth()
    trae_model = _map_model(model)
    body = _build_chat_body(messages, trae_model, stream)
    headers = _build_headers(token, user_id)

    last_error: Exception | None = None
    for endpoint in ENDPOINTS:
        url = base_url + endpoint
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = resp.read().decode("utf-8", errors="replace")
                log.info("Trae endpoint %s OK (status=%s)", endpoint, resp.status)
                return data
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:500]
            log.warning("Trae endpoint %s failed %s: %s", endpoint, e.code, detail)
            if e.code in (401, 403):
                raise HTTPException(status_code=401, detail=f"trae auth failed: {detail}")
            last_error = RuntimeError(f"{endpoint}: {e.code} {detail}")
        except Exception as e:
            log.warning("Trae endpoint %s error: %s", endpoint, e)
            last_error = e
    raise HTTPException(status_code=502, detail=f"trae all endpoints failed: {last_error}")

def _send_trae_work_chat(
    messages: list[dict[str, Any]],
    model: str,
    stream: bool,
    work: dict[str, Any],
) -> str:
    """Trae Work 通道：function=solo_work_lite + 完整 SOLO headers。

    参考 traework2api（Go）的 Work 通道实现：
    - headers 必须带 User-Agent: Trae/<ver> + X-Ide-Token 等 SOLO 专属头
      （缺 UA 会被服务端当异常客户端限流 4011）
    - host 用 mchost.guru（AgentHost），签到/积分才用 api.trae.cn
    """
    trae_model = _map_model(model)
    body = _build_chat_body(messages, trae_model, stream)
    # 绝大多数模型走 solo_work_lite；少数（glm-5.1 / Doubao-Seed-Code 等）
    # 在该 function 下 4001，需改走 chat_v3（实测）
    body["function"] = _WORK_FUNCTION_OVERRIDE.get(trae_model, "solo_work_lite")

    headers = {
        **_work_headers(work),
        "Accept": "text/event-stream" if stream else "application/json",
    }

    url = f"{BASE_URL_CN}/api/agent/v3/llm_utils_chat"
    # 上游断流多为瞬态（实测：长响应读到一半连接被关 → http.client.IncompleteRead，
    # 曾被流式层当正文吐成 "[Error: IncompleteRead(...)]"）。读失败后丢弃半截响应，
    # 同端点退避重试，避免一次网络抖动作废整轮长生成。
    payload = json.dumps(body).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(1, _WORK_CHAT_MAX_ATTEMPTS + 1):
        req = urllib.request.Request(
            url,
            data=payload,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                _debug_dump(
                    "debug_trae_upstream",
                    model=trae_model,
                    function=body["function"],
                    raw=raw[:20000],
                )
                return raw
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:500]
            raise HTTPException(status_code=502, detail=f"trae work chat failed: {e.code} {detail}")
        except Exception as e:
            # HTTPError 之外一律视为瞬态网络错误（IncompleteRead / 连接重置 / 超时等），
            # 保留原始异常重试；非瞬态错误重试 3 次也会快速失败，代价可接受。
            last_error = e
            if attempt < _WORK_CHAT_MAX_ATTEMPTS:
                log.warning(
                    "Trae Work chat attempt %d/%d failed (%s: %s), retrying",
                    attempt, _WORK_CHAT_MAX_ATTEMPTS, type(e).__name__, e,
                )
                time.sleep(attempt)
            else:
                log.error(
                    "Trae Work chat failed after %d attempts: %s: %s",
                    _WORK_CHAT_MAX_ATTEMPTS, type(e).__name__, e,
                )
    raise HTTPException(
        status_code=502,
        detail=f"trae work chat failed after {_WORK_CHAT_MAX_ATTEMPTS} attempts: {last_error}",
    )

