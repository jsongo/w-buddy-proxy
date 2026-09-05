"""原生 function calling 通道（chat_v3 / llm_raw_chat_v2 直通）。

2026-09 实测：llm_utils_chat 本身就是 raw-chat，原生支持 OpenAI 风格 tools
（parameters 需 JSON 字符串），响应 event:output 直接带按 index 增量合并的
tool_calls，多轮历史可结构化回放。上游按版本头门控模型能力，4001 拒绝时由
调用方回落文本协议（text_protocol）。
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
import uuid
from typing import Any

from fastapi import HTTPException

from .config import (
    BASE_URL_CN,
    _NATIVE_FUNCTION,
    _WORK_CHAT_MAX_ATTEMPTS,
    _map_model,
)
from .credentials import _auth, _build_headers, _load_work_cred, _work_headers
from .sse import _parse_sse
from .text_protocol import _content_blocks

log = logging.getLogger(__name__)

# ───────────────────────── 原生 function calling 通道 ─────────────────────────
#
# 2026-09 实测：llm_utils_chat 本身就是 raw-chat（chat_v3 的 timing 事件名即
# llm_raw_chat_v2），原生支持 OpenAI 风格 tools。此前走"提示词教学 + 文本解析"
# 是因为版本头停在 20260401，新一代模型全被上游按版本门控拒成 4001，误判为
# "上游不支持原生 tools 参数"。版本头提上去之后：
#   - tools 形状：{type:"function", function:{name, description, parameters}}，
#     parameters 必须是 JSON **字符串**（Go schema 反序列化错误实测确认）；
#   - 响应流 event:output 直接带 tool_calls：[{index, id, type:"function",
#     function_call:{name, arguments}}]，按 index 增量合并（OpenAI delta 风格，
#     字段名是 function_call 不是 function）；reasoning_content 独立字段；
#   - 多轮历史可结构化回放：assistant 带 tool_calls、role:"tool" 带
#     tool_call_id，上游原生认识（实测模型正确读取工具结果续答）；
#   - 覆盖面：17 个模型 16 个原生可用（唯一例外 glm-5-turbo 不在 chat_v3
#     通道，靠下方 4001 兜底自动回落文本协议）；
#   - tool_choice：只收字符串（auto/required/none/工具名都过校验），但语义
#     未验证（'none' 下模型仍会调工具），因此暂不透传。

def _native_rejected(raw: str) -> bool:
    """判断原生通道响应是否为上游参数拒绝（应回落文本协议）。

    拒绝有两种壳：HTTP 400 + JSON 体（{"code":4001,...}），或 HTTP 200 +
    SSE error 事件（event:error / data:{"code":4001,...}）。二者都统一由
    本函数识别。其他错误码（鉴权 1001、限流 4011 等）不回落——文本协议
    同样会失败，原样上报更有诊断价值。
    """
    if not raw:
        return False
    if raw.lstrip().startswith("{"):
        try:
            return str(json.loads(raw).get("code")) == "4001"
        except Exception:
            return False
    try:
        for event, data in _parse_sse(raw):
            if event == "error" and str((data or {}).get("code")) == "4001":
                return True
    except Exception:
        return False
    return False


def _native_tools_payload(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """OpenAI tools 定义 -> 原生通道 tools 数组（parameters 序列化成 JSON 字符串）。"""
    out: list[dict[str, Any]] = []
    for t in tools or []:
        if not isinstance(t, dict) or not t:
            continue
        fn = t["function"] if isinstance(t.get("function"), dict) else t
        name = fn.get("name") or t.get("name")
        if not name:
            continue
        params = fn.get("parameters") or t.get("parameters") or {"type": "object", "properties": {}}
        if not isinstance(params, str):
            params = json.dumps(params, ensure_ascii=False)
        out.append({
            "type": "function",
            "function": {
                "name": name,
                "description": fn.get("description") or t.get("description") or "",
                "parameters": params,
            },
        })
    return out


def _native_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenAI messages -> 原生通道 messages（结构化历史，不做文本协议改写）。

    与 _extract_prompt 的关键差异：assistant.tool_calls 保持结构化
    （function_call 形状，2026-09 实测上游原生认识），role:tool 结果保持
    tool 角色 + tool_call_id，不转 user + [tool_result] 前缀。
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        role = m.get("role", "user")
        content = _content_blocks(m.get("content"))
        tool_calls = m.get("tool_calls") if isinstance(m, dict) else None
        if role == "assistant" and tool_calls:
            tcs: list[dict[str, Any]] = []
            for j, c in enumerate(tool_calls):
                fn = c.get("function") if isinstance(c.get("function"), dict) else c
                args = fn.get("arguments", "")
                if not isinstance(args, str):
                    args = json.dumps(args or {}, ensure_ascii=False)
                tcs.append({
                    "index": j,
                    "id": c.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function_call": {"name": fn.get("name") or "", "arguments": args},
                })
            # content 为空也要带空 text block（上游 schema 要求；实测空串即可）
            out.append({"role": "assistant", "content": content, "tool_calls": tcs})
        elif role in ("tool", "function"):
            out.append({
                "role": "tool",
                "tool_call_id": m.get("tool_call_id") or "",
                "name": m.get("name") or "tool",
                "content": content,
            })
        else:
            out.append({"role": role, "content": content})
    return out


def _build_native_body(
    native_msgs: list[dict[str, Any]], trae_model: str, stream: bool,
    tools: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    session_id = str(uuid.uuid4())
    return {
        "messages": native_msgs,
        "model": trae_model,
        "config_name": trae_model,
        "function": _NATIVE_FUNCTION,
        "stream": stream,
        "request_id": session_id,
        "session_id": session_id,
        "tools": _native_tools_payload(tools),
    }


def _send_native_chat(
    native_msgs: list[dict[str, Any]], model: str, stream: bool,
    tools: list[dict[str, Any]] | None,
) -> str:
    """原生通道发送：Work 凭证优先，回退 IDE 凭证；返回原始 SSE/错误文本。

    HTTP 400（Go schema 拒绝的另一种壳）原样返回文本，交给调用方
    _native_rejected 判定回落，不在这里抛——文本协议兜底是对 4001 的
    正确响应，抛异常会跳过兜底。
    """
    trae_model = _map_model(model)
    body = _build_native_body(native_msgs, trae_model, stream, tools)
    work = _load_work_cred()
    if work and work.get("access_token"):
        headers = _work_headers(work)
    else:
        token, user_id = _auth()
        headers = _build_headers(token, user_id)
    headers = {
        **headers,
        "Accept": "text/event-stream" if stream else "application/json",
    }
    url = f"{BASE_URL_CN}/api/agent/v3/llm_utils_chat"
    payload = json.dumps(body).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(1, _WORK_CHAT_MAX_ATTEMPTS + 1):
        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:2000]
            if e.code == 400:
                # Go schema 拒绝的另一种壳：原样返回文本，交给 _native_rejected
                # 判定回落（文本协议兜底是对 4001 的正确响应）
                return detail or '{"code":4001}'
            raise HTTPException(status_code=502, detail=f"trae native chat failed: {e.code} {detail}")
        except Exception as e:
            # 与 Work 通道同款：HTTPError 之外视为瞬态（IncompleteRead/断连/超时），
            # 同端点退避重试；4xx 参数拒绝不走这里（上面已返回/抛出）
            last_error = e
            if attempt < _WORK_CHAT_MAX_ATTEMPTS:
                log.warning(
                    "Trae native chat attempt %d/%d failed (%s: %s), retrying",
                    attempt, _WORK_CHAT_MAX_ATTEMPTS, type(e).__name__, e,
                )
                time.sleep(attempt)
    raise HTTPException(
        status_code=502,
        detail=f"trae native chat failed after {_WORK_CHAT_MAX_ATTEMPTS} attempts: {last_error}",
    )


class _NativeToolAccumulator:
    """按 index 合并原生通道流式 tool_calls 分片。

    上游分片语义（实测）：首事件可能携带完整调用（id/name/arguments 一次给全），
    后续事件按 index 追加——新调用以"非空 id"宣告，name 随后的非空分片到达，
    arguments 非空分片为**追加**而非替换（观测到 '{"c' 这样的残片）。空串
    字段一律忽略。
    """

    def __init__(self) -> None:
        self._calls: dict[int, dict[str, str]] = {}

    def feed(self, items: Any) -> None:
        for tc in items or []:
            if not isinstance(tc, dict):
                continue
            try:
                idx = int(tc.get("index") or 0)
            except (TypeError, ValueError):
                idx = 0
            fc = tc.get("function_call") if isinstance(tc.get("function_call"), dict) else {}
            call = self._calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
            if tc.get("id"):
                call["id"] = tc["id"]
            if fc.get("name"):
                call["name"] = fc["name"]
            if fc.get("arguments"):
                call["arguments"] += fc["arguments"]

    def finish(self) -> list[dict[str, Any]]:
        """输出 OpenAI chat.completion 形状的 tool_calls（含 index，按序排列）。"""
        return [
            {
                "index": i,
                "id": c["id"] or f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {"name": c["name"], "arguments": c["arguments"]},
            }
            for i, c in sorted(self._calls.items())
        ]
