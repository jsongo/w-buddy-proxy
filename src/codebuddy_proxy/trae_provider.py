"""Trae Provider —— 直连 Trae IDE 底层模型，转成 OpenAI 兼容格式。

从 trae-local-api（Node.js 版，https://github.com/numbqr123/trae-local-api）抽取
核心逻辑翻译成 Python：

1. **认证**：解密 Trae IDE 本地存储的 tc 加密格式（AES-128-CBC + SHA-512），
   拿到 Cloud-IDE-JWT token；或从 .env 读已解密的 token。
2. **调用**：直连 ``https://trae-api-cn.mchost.guru/api/agent/v3/llm_utils_chat``
   （3 级端点回退），SSE 流式返回。
3. **转换**：Trae SSE（``event: output`` + ``{response, reasoning_content}``）
   → OpenAI chat.completion 格式（流式 / 非流式）。

已知问题：Trae 上游模型层可能返回 ``3003 all models failed``（账号/配额/
服务端问题），此时会透传错误信息给调用方。

依赖：纯标准库（urllib + hashlib），零第三方依赖。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import sys
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Sequence

from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from .providers import BaseProvider

log = logging.getLogger(__name__)

# ───────────────────────── Trae 常量 ─────────────────────────

BASE_URL_CN = "https://trae-api-cn.mchost.guru"
BASE_URL_SG = "https://a0ai-api-sg.byteintlapi.com"
IDE_VERSION = "3.3.67"
IDE_VERSION_CODE = "20260401"
X_APP_ID = "6eefa01c-1036-4c7e-9ca5-d891f63bfcd8"

# 3 级端点回退（与 trae-local-api 一致）
ENDPOINTS = [
    "/api/agent/v3/llm_utils_chat",
    "/api/ide/v1/chat",
    "/api/agent/v3/create_agent_task",
]

# 模型名映射：外部名 -> Trae 内部名（与 trae-local-api 一致）
MODEL_MAP: dict[str, str] = {
    "claude-opus-4-7": "glm-5.2",
    "claude-opus-4-6": "glm-5.2",
    "claude-opus-4-5": "glm-5.2",
    "claude-sonnet-4-6": "glm-5.2",
    "claude-sonnet-4-5": "glm-5.2",
    "claude-sonnet-4": "glm-5.2",
    "claude-3.5-sonnet": "glm-5.2",
    "claude-3.7-sonnet": "glm-5.2",
    "claude-haiku-4-5": "glm-5.1",
    "mimo-v2.5-pro": "glm-5.2",
    "mimo-v2.5": "glm-5.2",
    "gpt-4o": "DeepSeek-V4-Pro",
    "gpt-4o-mini": "DeepSeek-V4-Flash",
    "gpt-4.1": "DeepSeek-V4-Pro",
    "auto": "glm-5.2",
}

# 模型分级（T1 最强 -> T5 最弱）
MODEL_TIERS: dict[str, list[str]] = {
    "T1": ["glm-5.2"],
    "T2": ["glm-5.1", "qwen-3.7-plus", "kimi-k2.6", "DeepSeek-V4-Pro"],
    "T3": ["glm-5", "qwen-3.6-plus", "minimax-m3", "DeepSeek-V4-Flash"],
    "T4": ["glm-4.7", "kimi-k2", "qwen3-coder", "minimax-m2.7"],
    "T5": ["glm-4.6", "minimax-m2.1"],
}


# ───────────────────────── tc 加密解密 ─────────────────────────

# 4 组 64 字节 salt（从 Trae CN 前端 JS 提取）
_SALT_A = bytes([
    82, 9, 106, 213, 48, 54, 165, 56, 191, 64, 163, 158, 129, 243, 215, 251,
    124, 227, 57, 130, 155, 47, 255, 135, 52, 142, 67, 68, 196, 222, 233, 203,
    84, 123, 148, 50, 166, 194, 35, 61, 238, 76, 149, 11, 66, 250, 195, 78,
    8, 46, 161, 102, 40, 217, 36, 178, 118, 91, 162, 73, 109, 139, 209, 37,
])
_SALT_B = bytes([
    31, 221, 168, 51, 136, 7, 199, 49, 177, 18, 16, 89, 39, 128, 236, 95,
    96, 81, 127, 169, 25, 181, 74, 13, 45, 229, 122, 159, 147, 201, 156, 239,
    160, 224, 59, 77, 174, 42, 245, 176, 200, 235, 187, 60, 131, 83, 153, 97,
    23, 43, 4, 126, 186, 119, 214, 38, 225, 105, 20, 99, 85, 33, 12, 125,
])
_SALT_C = bytes([
    191, 192, 216, 250, 122, 246, 220, 97, 31, 254, 98, 27, 8, 72, 71, 176,
    135, 99, 96, 18, 127, 101, 203, 104, 211, 102, 191, 125, 37, 72, 150, 156,
    51, 229, 121, 35, 17, 153, 141, 177, 110, 131, 150, 128, 172, 255, 254, 6,
    18, 140, 55, 62, 236, 249, 135, 64, 135, 12, 117, 4, 89, 149, 168, 209,
])
_SALT_D = bytes([
    246, 204, 26, 232, 232, 70, 129, 109, 223, 146, 169, 242, 23, 241, 105, 145,
    50, 196, 165, 42, 254, 120, 3, 54, 244, 207, 209, 85, 53, 6, 138, 106,
    175, 148, 31, 204, 186, 186, 165, 182, 87, 142, 49, 10, 39, 110, 26, 154,
    86, 56, 173, 125, 18, 64, 198, 225, 99, 99, 83, 82, 191, 134, 76, 170,
])


def _xor_bytes(a: bytes, b: bytes) -> bytes:
    return bytes(x ^ y for x, y in zip(a, b))


def _aes_cbc_decrypt(key: bytes, iv: bytes, data: bytes) -> bytes:
    """AES-128-CBC 解密（纯 Python 实现，零依赖）。"""
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        decryptor = cipher.decryptor()
        return decryptor.update(data) + decryptor.finalize()
    except ImportError:
        return _aes_cbc_decrypt_pure(key, iv, data)


def _aes_cbc_decrypt_pure(key: bytes, iv: bytes, data: bytes) -> bytes:
    """纯 Python AES-128-CBC（无 cryptography 时的兜底）。"""
    from .aes_pure import aes_cbc_decrypt

    return aes_cbc_decrypt(key, iv, data)


def decrypt_storage_value(base64_value: str) -> str:
    """解密单个 tc 格式加密值。

    结构: [6B Header][32B RandomBytes][N EncryptedData]
    解密: [64B SHA-512 Hash][N Plaintext JSON]
    """
    import base64

    buf = base64.b64decode(base64_value)
    header = buf[:6]
    random_bytes = buf[6:38]
    encrypted = buf[38:]

    # 检测加密类型
    if header[:2] == b"\x74\x63" and header[2:6] == b"\x05\x10\x00\x00":
        salt = _xor_bytes(_SALT_A, _SALT_B)
    elif header[:2] == b"\x12\x39\x20\x20\x02\x03":
        salt = _xor_bytes(_SALT_C, _SALT_D)
    else:
        raise ValueError(f"未知加密类型: {header.hex()}")

    # SHA-512(RandomBytes) -> hashOfRandom
    hash_of_random = hashlib.sha512(random_bytes).digest()
    # SHA-512(hashOfRandom + salt) -> finalHash
    final_hash = hashlib.sha512(hash_of_random + salt).digest()
    aes_key = final_hash[:16]
    iv = final_hash[16:32]

    # AES-128-CBC 解密
    decrypted = _aes_cbc_decrypt(aes_key, iv, encrypted)

    # 校验: 前 64B 是 SHA-512 hash
    stored_hash = decrypted[:64]
    plaintext = decrypted[64:]
    computed = hashlib.sha512(plaintext).digest()
    if stored_hash != computed:
        raise ValueError("hash 校验失败，解密可能不正确")
    return plaintext.decode("utf-8")


def decrypt_auth_data(data_dir: str) -> dict[str, Any]:
    """从 Trae 的 storage.json 解密认证数据。

    Args:
        data_dir: Trae 用户数据目录（如 ~/Library/Application Support/Trae CN/User）
    """
    storage_path = Path(data_dir) / "globalStorage" / "storage.json"
    if not storage_path.exists():
        raise FileNotFoundError(f"storage.json not found: {storage_path}")

    storage = json.loads(storage_path.read_text("utf-8"))
    encrypted = storage.get("iCubeAuthInfo://icube.cloudide")
    if not encrypted:
        raise KeyError("iCubeAuthInfo://icube.cloudide key not found in storage.json")

    # SG 版是明文 JSON
    if encrypted.strip().startswith("{"):
        return json.loads(encrypted)

    # CN 版 tc 加密
    decrypted = decrypt_storage_value(encrypted)
    return json.loads(decrypted)


def _app_data_root() -> Path:
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support"
    return Path.home() / ".config"


def find_auth_data(edition: str = "cn") -> dict[str, Any]:
    """自动查找并解密 Trae 认证数据（支持 CN/SG/SOLO）。"""
    root = _app_data_root()
    candidates = {
        "cn": [root / "Trae CN" / "User", root / "Trae" / "User"],
        "solo": [root / "Trae SOLO" / "User"],
        "solo-sg": [root / "Trae SOLO" / "User"],
        "sg": [root / "Trae" / "User"],
    }
    for dirpath in candidates.get(edition, candidates["cn"]):
        try:
            data = decrypt_auth_data(str(dirpath))
            log.info("Trae auth decrypted from %s", dirpath)
            return data
        except Exception as e:
            log.debug("Trae decrypt failed for %s: %s", dirpath, e)
    raise RuntimeError(f"无法解密 Trae {edition} 认证数据")


def _extract_auth_fields(data: dict[str, Any]) -> tuple[str, str]:
    """从解密数据提取 (token, userId)。"""
    # 常见字段名（不同版本存储结构略有差异）
    token = (
        data.get("token")
        or data.get("Token")
        or data.get("access_token")
        or (data.get("data") or {}).get("token")
        or ""
    )
    user_id = (
        data.get("userId")
        or data.get("UserID")
        or data.get("user_id")
        or data.get("uid")
        or ""
    )
    return str(token), str(user_id)


# ───────────────────────── Trae API 调用 ─────────────────────────

def _build_headers(token: str, user_id: str) -> dict[str, str]:
    machine_id = uuid.uuid4().hex
    device_id = hashlib.sha256(machine_id.encode()).hexdigest()[:32]
    return {
        "Authorization": f"Cloud-IDE-JWT {token}",
        "X-Cloudide-Token": token,
        "x-uid": user_id or "",
        "x-app-id": X_APP_ID,
        "x-device-id": device_id,
        "x-machine-id": machine_id,
        "x-request-id": str(uuid.uuid4()),
        "x-ide-version": IDE_VERSION,
        "x-ide-version-code": IDE_VERSION_CODE,
        "x-device-type": "windows",
        "x-os-version": "Windows 10",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }


def _map_model(requested: str) -> str:
    return MODEL_MAP.get(requested, requested)


def _extract_prompt(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenAI messages -> Trae messages（content 转成 block 数组）。"""
    out = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "".join(
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        else:
            text = ""
        out.append({"role": role, "content": [{"type": "text", "text": text}]})
    return out


def _build_chat_body(messages: list[dict[str, Any]], model: str, stream: bool) -> dict[str, Any]:
    session_id = str(uuid.uuid4())
    return {
        "messages": messages,
        "model": model,
        "function": "inline_chat",
        "stream": stream,
        "request_id": session_id,
        "session_id": session_id,
    }


def send_trae_chat(
    messages: list[dict[str, Any]],
    model: str,
    stream: bool,
    base_url: str = BASE_URL_CN,
) -> str:
    """直连 Trae API，返回原始 SSE 文本。

    3 级端点回退。认证失败抛 HTTPException(401)，其余抛 RuntimeError。
    """
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


# 全局认证缓存（惰性加载）
_auth_cache: tuple[str, str] | None = None


def _auth() -> tuple[str, str]:
    """获取 (token, user_id)：优先 .env，其次解密 storage.json。"""
    global _auth_cache
    if _auth_cache:
        return _auth_cache

    # 1) 环境变量 / .env
    token = os.environ.get("TRAE_TOKEN", "")
    user_id = os.environ.get("TRAE_USER_ID", "")
    if not token:
        # 尝试 trae-local-api 的 .env
        env_path = Path.home() / "code" / "others" / "trae-local-api" / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line.startswith("TRAE_TOKEN="):
                    token = line.split("=", 1)[1]
                elif line.startswith("TRAE_USER_ID="):
                    user_id = line.split("=", 1)[1]
    if token:
        _auth_cache = (token, user_id)
        return _auth_cache

    # 2) 解密 Trae 本地存储
    for edition in ("cn", "sg", "solo"):
        try:
            data = find_auth_data(edition)
            token, user_id = _extract_auth_fields(data)
            if token:
                _auth_cache = (token, user_id)
                log.info("Trae auth loaded via %s decrypt", edition)
                return _auth_cache
        except Exception as e:
            log.debug("Trae %s auth failed: %s", edition, e)

    raise HTTPException(status_code=401, detail="trae not authenticated - 请先登录 Trae IDE")


# ───────────────────────── SSE 解析 ─────────────────────────

def _parse_sse(text: str) -> list[tuple[str, dict[str, Any]]]:
    """解析 Trae SSE 流 -> [(event, data_dict), ...]。"""
    events = []
    current_event = ""
    current_data: list[str] = []

    def flush():
        if current_data:
            data_str = "\n".join(current_data)
            try:
                parsed = json.loads(data_str)
            except Exception:
                parsed = {"raw": data_str}
            events.append((current_event, parsed))
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


# ───────────────────────── Provider 实现 ─────────────────────────

class TraeProvider(BaseProvider):
    id = "trae"
    name = "Trae (本地解密直连)"

    def __init__(self, base_url: str | None = None, edition: str = "cn"):
        self._base_url = base_url or BASE_URL_CN
        self._edition = edition

    def models(self) -> Sequence[dict[str, Any]]:
        result = []
        seen = set()
        for tier, models in MODEL_TIERS.items():
            for m in models:
                if m in seen:
                    continue
                seen.add(m)
                result.append({
                    "id": m,
                    "object": "model",
                    "created": 0,
                    "owned_by": self.id,
                    "tier": tier,
                    "description": f"Trae {tier} 模型",
                })
        # 加别名（外部名映射）
        for external, internal in MODEL_MAP.items():
            if external not in seen:
                seen.add(external)
                result.append({
                    "id": external,
                    "object": "model",
                    "created": 0,
                    "owned_by": self.id,
                    "maps_to": internal,
                    "description": f"Trae 别名 -> {internal}",
                })
        return result

    def ensure_auth(self) -> None:
        _auth()

    async def forward(
        self,
        body: dict[str, Any],
        protocol: str,
        original: dict[str, Any] | None = None,
    ) -> StreamingResponse | JSONResponse:
        requested_model = body.get("model", "auto")
        messages = body.get("messages", [])
        stream = bool(body.get("stream", False))

        prompt = _extract_prompt(messages)
        if not prompt:
            raise HTTPException(status_code=400, detail="no text content")

        if stream:
            return StreamingResponse(
                self._stream(prompt, requested_model),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "close"},
            )
        return JSONResponse(content=self._collect(prompt, requested_model))

    def _stream(self, messages: list[dict[str, Any]], model: str) -> AsyncIterator[str]:
        """把 Trae SSE 转成 OpenAI chat.completion.chunk 流。"""
        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        in_thinking = False

        def chunk(delta: dict[str, Any], finish: str | None = None) -> str:
            payload = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        try:
            raw = send_trae_chat(messages, model, stream=True, base_url=self._base_url)
            for event, data in _parse_sse(raw):
                if event == "error":
                    msg = data.get("message") or data.get("error") or ""
                    yield chunk({"content": f"[Trae Error: {msg}]"})
                    yield "data: [DONE]\n\n"
                    return
                if event == "output":
                    if data.get("reasoning_content"):
                        in_thinking = True
                        yield chunk({"reasoning_content": data["reasoning_content"]})
                    if data.get("response"):
                        yield chunk({"role": "assistant", "content": data["response"]})
                        in_thinking = False
                elif event == "done":
                    break
        except HTTPException as e:
            yield chunk({"content": f"[Error {e.status_code}]"})
            yield "data: [DONE]\n\n"
            return
        except Exception as e:
            log.error("Trae stream error: %s", e)
            yield chunk({"content": f"[Error: {e}]"})

        yield chunk({}, "stop")
        yield "data: [DONE]\n\n"

    def _collect(self, messages: list[dict[str, Any]], model: str) -> dict[str, Any]:
        """非流式：聚合 Trae SSE 成完整响应。"""
        reasoning_parts: list[str] = []
        content_parts: list[str] = []

        raw = send_trae_chat(messages, model, stream=False, base_url=self._base_url)
        for event, data in _parse_sse(raw):
            if event == "error":
                raise HTTPException(
                    status_code=502,
                    detail=f"trae upstream error: {data.get('message') or data.get('error')}",
                )
            if event == "output":
                if data.get("reasoning_content"):
                    reasoning_parts.append(data["reasoning_content"])
                if data.get("response"):
                    content_parts.append(data["response"])

        reasoning = "".join(reasoning_parts)
        content = "".join(content_parts)
        if not content and not reasoning:
            content = "(trae upstream 返回了空响应，未产生任何内容)"
        if reasoning and content:
            content = f"<think>\n{reasoning}\n</think>\n\n{content}"

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": max(1, len(content.encode("utf-8")) // 4),
                "total_tokens": 0,
            },
        }
