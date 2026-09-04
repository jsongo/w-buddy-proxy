"""Trae Provider —— 直连 Trae IDE 底层模型，转成 OpenAI 兼容格式。

从 trae-local-api（Node.js 版，https://github.com/numbqr123/trae-local-api）抽取
核心逻辑翻译成 Python：

1. **认证**：解密 Trae IDE 本地存储的 tc 加密格式（AES-128-CBC + SHA-512），
   拿到 Cloud-IDE-JWT token；或从 .env 读已解密的 token。
2. **调用**：直连 ``https://trae-api-cn.mchost.guru/api/agent/v3/llm_utils_chat``
   （3 级端点回退），SSE 流式返回。
3. **转换**：Trae SSE（``event: output`` + ``{response, reasoning_content}``）
   → OpenAI chat.completion 格式（流式 / 非流式）。
   另支持 anthropic 协议（/v1/messages，Claude Code 等）：请求侧由
   anthropic_adapter 转成 chat，响应侧经 _wrap_anthropic_stream /
   chat_completion_to_anthropic_message 还原成 Anthropic 格式。

已知问题：Trae 上游模型层可能返回 ``3003 all models failed``（账号/配额/
服务端问题），此时会透传错误信息给调用方。

依赖：纯标准库（urllib + hashlib），零第三方依赖。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import platform
import re
import sys
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Iterator, Sequence

from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from .providers import BaseProvider
from .anthropic_adapter import (
    AnthropicStreamConverter,
    chat_completion_to_anthropic_message,
)

log = logging.getLogger(__name__)

# ───────────────────────── Trae 常量 ─────────────────────────

BASE_URL_CN = "https://trae-api-cn.mchost.guru"
BASE_URL_SG = "https://a0ai-api-sg.byteintlapi.com"
IDE_VERSION = "3.3.67"
IDE_VERSION_CODE = "20260401"
X_APP_ID = "6eefa01c-1036-4c7e-9ca5-d891f63bfcd8"

# Trae Work (SOLO) 客户端版本。
# 注意：服务端按版本号 gating 新模型——旧版本（0.1.43/20260716）请求 glm-5.3
# 等新模型会被拒为 4001 "param is invalid"；升级后才返回真实路由（实测）。
_TRAE_APP_VERSION = "0.2.0"
_TRAE_APP_VERSION_CODE = "20260901"

# 3 级端点回退（与 trae-local-api 一致）
ENDPOINTS = [
    "/api/agent/v3/llm_utils_chat",
    "/api/ide/v1/chat",
    "/api/agent/v3/create_agent_task",
]

# 模型名映射：外部别名 -> Trae 内部 config_name（大小写敏感，须与下方实测白名单一致）
MODEL_MAP: dict[str, str] = {
    # "claude-opus-4-7": "glm-5.3",
    # "claude-opus-4-6": "glm-5.3",
    # "claude-opus-4-5": "glm-5.3",
    # "claude-sonnet-4-6": "glm-5.3",
    # "claude-sonnet-4-5": "glm-5.3",
    # "claude-sonnet-4": "glm-5.3",
    # "claude-3.5-sonnet": "glm-5.3",
    # "claude-3.7-sonnet": "glm-5.3",
    # "claude-haiku-4-5": "DeepSeek-V4-Flash",
    # "mimo-v2.5-pro": "glm-5.3",
    # "mimo-v2.5": "glm-5.3",
    # "gpt-4o": "DeepSeek-V4-Pro",
    # "gpt-4o-mini": "DeepSeek-V4-Flash",
    # "gpt-4.1": "DeepSeek-V4-Pro",
    # "auto": "glm-5.3",
    # 大小写容错：Trae config_name 大小写不统一（DeepSeek-/Doubao- 为大写前缀），
    # 而 OpenAI 生态习惯全小写；小写请求若不在此映射，会落回默认 CodeBuddy 通道
    # （报错表现为 CodeBuddy 上游的安全审核/路由错误，而非 Trae 响应）
    "deepseek-v4-pro": "DeepSeek-V4-Pro",
    "deepseek-v4-flash": "DeepSeek-V4-Flash",
    "doubao-seed-evolving": "Doubao-Seed-Evolving",
    "doubao-seed-2.1-pro": "Doubao-Seed-2.1-Pro",
    "doubao-seed-2.1-turbo": "Doubao-Seed-2.1-Turbo",
    "doubao-seed-code": "Doubao-Seed-Code",
    # qwen 官方命名点号/连字符混用，两种写法都放行
    "qwen-3.8-max": "qwen3.8-max",
    "qwen3.7-plus": "qwen-3.7-plus",
}

# 模型分级（T1 最强 -> T4 最弱）。
# config_name 全部为 2026-09 实测通过值（Work 凭证 + llm_utils_chat 端点）：
# 注意命名大小写不统一——DeepSeek-/Doubao- 为大写前缀，glm/kimi/minimax 小写，
# qwen 两种写法并存（qwen3.8-max 用点号、qwen-3.7-plus 用连字符）。
# kimi-k3 虽在官方文档内，但需会员 Pro+/Ultra/Express（免费账号 1005），故不列出。
MODEL_TIERS: dict[str, list[str]] = {
    "T1": ["glm-5.3", "glm-5.3-flash", "Doubao-Seed-Evolving"],
    "T2": ["glm-5.2", "Doubao-Seed-2.1-Pro", "DeepSeek-V4-Pro",
           "kimi-k2.7-code", "qwen3.8-max"],
    "T3": ["Doubao-Seed-2.1-Turbo", "DeepSeek-V4-Flash", "minimax-m3",
           "kimi-k2.6", "glm-5.1"],
    "T4": ["Doubao-Seed-Code", "glm-5", "glm-5-turbo", "qwen-3.7-plus"],
}

# 部分模型在 solo_work_lite function 下不可用（服务端 4001），需改用 chat_v3。
# 实测（2026-09-03）：glm-5.1 / Doubao-Seed-Code 仅在 chat_v3 下可路由；
# glm-5.3-flash 同理——solo_work_lite 报 4001 "param is invalid"，
# chat_v3 正常出流（会话 s_20260903_2108_4190 即踩此坑）。
_WORK_FUNCTION_OVERRIDE: dict[str, str] = {
    "glm-5.1": "chat_v3",
    "Doubao-Seed-Code": "chat_v3",
    "glm-5.3-flash": "chat_v3",
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
    elif header[:6] == b"\x12\x39\x20\x20\x02\x03":
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


# ───────────────────────── 签到 / 积分 ─────────────────────────

_UG_API_HOST = "https://api.trae.cn"
# 签到 API 是 device 维度的：device_id 从 JWT 里的稳定 userId 派生（trae2api-cn 方案）
_CHECKIN_DEVICE_IDS: dict[str, str] = {}


def _checkin_identity(token: str, account_id: str = "") -> str:
    """从 JWT 提取稳定 identity（不依赖可能刷新的 token 本身）。"""
    if token:
        try:
            parts = token.split(".")
            if len(parts) >= 2:
                import base64 as _b64

                encoded = parts[1] + "=" * (-len(parts[1]) % 4)
                payload = json.loads(_b64.urlsafe_b64decode(encoded.encode("ascii")))
                data = payload.get("data")
                if isinstance(data, dict) and data.get("id"):
                    return str(data["id"])
                for key in ("user_id", "userId", "sub"):
                    if payload.get(key):
                        return str(payload[key])
        except Exception:
            pass
    if account_id:
        return str(account_id)
    return token


def checkin_device_id(token: str, account_id: str = "") -> str:
    """返回账号绑定的 16 位稳定 device id（签到 API 需要）。"""
    identity = _checkin_identity(token, account_id)
    if not identity:
        return ""
    cache_key = f"checkin#{identity}"
    if cache_key in _CHECKIN_DEVICE_IDS:
        return _CHECKIN_DEVICE_IDS[cache_key]
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    did = str(int(digest, 16) % 10**16).zfill(16)
    _CHECKIN_DEVICE_IDS[cache_key] = did
    return did


def _build_checkin_headers(token: str, account_id: str = "") -> dict[str, str]:
    headers = {
        "Authorization": f"Cloud-IDE-JWT {token}",
        "Content-Type": "application/json",
        "x-device-id": checkin_device_id(token, account_id),
        "x-device-brand": "ASUS TUF Gaming A15 FA507RM_FA507RM",
        "x-device-type": "windows",
    }
    return headers


def _post_ug(path: str, token: str = "", account_id: str = "") -> dict[str, Any]:
    """调用 Trae UG（user growth）签到/积分 API。"""
    if not token:
        token, _ = _auth()
    url = _UG_API_HOST + path
    req = urllib.request.Request(
        url,
        data=b"{}",
        headers=_build_checkin_headers(token, account_id),
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Trae UG {path} [{e.code}]: {e.read().decode()[:300]}")
    return data


def fetch_checkin_status(token: str = "", account_id: str = "") -> dict[str, Any]:
    """查询今日签到/积分状态。"""
    return _post_ug("/trae/api/v2/ug/checkin_credits/status", token, account_id)


def claim_checkin_credits(token: str = "", account_id: str = "") -> dict[str, Any]:
    """领取今日签到积分。"""
    return _post_ug("/trae/api/v2/ug/checkin_credits/claim", token, account_id)


# ───────────────────────── Trae API 调用 ─────────────────────────

def _build_headers(token: str, user_id: str) -> dict[str, str]:
    """构建 SOLO 完整请求头（traework2api headers.go 实测值）。

    关键：必须带 User-Agent: Trae/<ver> + X-Ide-Token 等 SOLO 专属头，
    缺 UA 会被服务端当异常客户端限流（4011）。
    """
    machine_id = uuid.uuid4().hex
    device_id = hashlib.sha256(machine_id.encode()).hexdigest()[:32]
    return {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "User-Agent": f"Trae/{_TRAE_APP_VERSION}",
        "Authorization": f"Cloud-IDE-JWT {token}",
        "X-Cloudide-Token": token,
        "X-Ide-Token": token,
        "X-Uid": user_id or "",
        "X-App-Id": X_APP_ID,
        "X-App-Version": "default",
        "X-Ide-Version": _TRAE_APP_VERSION,
        "X-Ide-Version-Code": _TRAE_APP_VERSION_CODE,
        "X-App-Version-Code": _TRAE_APP_VERSION_CODE,
        "X-Ide-Version-Type": "stable",
        "X-Device-Type": "windows",
        "X-OS-Version": "Windows 11 Pro",
        "X-Device-Brand": "83DG",
        "Request-Traffic-Type": "prod",
        "X-Machine-Id": machine_id,
        "X-Device-Id": device_id,
    }


def _map_model(requested: str) -> str:
    return MODEL_MAP.get(requested, requested)


def _debug_dump(event: str, **kwargs: Any) -> None:
    """WB_DEBUG_DUMP=1 时把调试事件写进 jsonl 日志（完整请求/响应/路由决策）。

    惰性 import state 避免模块级循环依赖；失败静默（调试设施不能影响主流程）。
    """
    if not os.environ.get("WB_DEBUG_DUMP"):
        return
    try:
        from .state import get_state
        get_state().write_log(event, **kwargs)
    except Exception:
        pass


# 注入的 agent 协议压制指令（见 _extract_prompt 说明）
_AGENT_GUARD = (
    "当前环境是纯文本对话 API，没有任何命令执行、文件读写等工具，也没有 shell。"
    "禁止输出任何工具调用标记（如 <Command>、<tool_action>、execute_command 等"
    "XML 标签或伪协议块）。如问题看似需要执行命令/操作系统，直接用自然语言给出"
    "回答或方案，示例命令用 markdown 代码块展示。"
)

# 输出侧兜底清洗：即使 system 注入失效，也把漏出的 agent 协议标记剥掉。
# 实测泄漏格式（Trae agent 端点的服务端预设导致模型输出工具调用语法）：
#   <Command len=22 id=0 run=1369463>\nlsof ...\n</Command>
#   以及 tool_call 残留前缀、tool_action 块、think 标签、错误提示行。
# 注意 <Command\\s 要求标签名后跟空白接属性（len=/id=/run=），避免误伤
# <CommandBuffer> 这类正常技术词汇；正文中单独的 Command 单词不会匹配。
_LEAK_RE = [
    # 带属性签名的命令块（len=/id=，可带 tool_call 残留前缀；
    # 上游泄漏格式可能残缺：< 不总存在、截断流可能没有 >）
    re.compile(r"(?:</?\s*tool_call>\s*)?<?\s*Command\s+len=\d+\s+id=\d+[^>\n]*>?.*?</Command>", re.S),
    re.compile(r"(?:</?\s*tool_call>\s*)?<?\s*Command\s+len=\d+\s+id=\d+[^>\n]*>?.*", re.S),  # 未闭合尾巴（截断流）
    # 常规标签形态的命令块（backup）
    re.compile(r"<Command\s[^>]*>.*?</Command>", re.S),
    re.compile(r"<Command\s[^>]*>.*", re.S),
    re.compile(r"<Command\s[^>]*>"),
    re.compile(r"</Command>"),
    # tool_action 完整块与孤立标签
    re.compile(r"<tool_action[^>]*>.*?</tool_action>", re.S),
    re.compile(r"</?\s*tool_action[^>]*>"),
    # 孤立的 tool_name / command 块（tool_action 外壳被剥离/缺失时，
    # 内部标签会裸露——实测下游 coding agent 收到的正是这种残骸）
    re.compile(r"<tool_name[^>]*>.*?</tool_name>", re.S),
    re.compile(r"</?\s*tool_name[^>]*>"),
    re.compile(r"<command>.*?</command>", re.S | re.I),
    re.compile(r"</?\s*command>", re.I),
    # think 块（reasoning 泄漏到正文）与孤立 think 标签
    re.compile(r"<think>.*?</think>", re.S),
    re.compile(r"</?\s*think>"),
    # 伪回调头 <tool_callback>（deepseek-v4-flash 实测：开标签后跟意图文本、
    # 不闭合也不携带调用，纯粹是协议噪音）。剥法与 _parse_tool_calls 统一：
    # 闭合块整块剥；未闭合但后面跟 <tool_call> 的，剥到调用前（真调用交给
    # 后续/解析层）；真到流尾都没闭合的（模型放弃调用继续说正文），只剥
    # 标签壳、保留内部文本——否则伪头之后的正文会被整段吞掉。
    # 必须排在孤立 tool_call 模式之前——tool_call[^>]* 会把 <tool_callback>
    # 误当孤儿 tool_call 标签先剥壳，导致块模式失配、内部文本泄漏
    re.compile(r"<tool_callback[^>]*>.*?</tool_callback>", re.S),
    re.compile(r"<tool_callback[^>]*>.*?(?=<tool_call[>\s])", re.S),
    re.compile(r"</?\s*tool_callback[^>]*>"),
    # 孤立 tool_call 标签
    re.compile(r"</?\s*tool_call[^>]*>"),
    # 孤立 arg_key / arg_value 标签（工具调用残骸，glm-5.3 实测）
    re.compile(r"</?\s*arg_(?:key|value)[^>]*>"),
    # 工具执行失败的提示行
    re.compile(r"执行过程发生错误[:：][^\n]*"),
]


def _sanitize_agent_leak(text: str) -> str:
    """剥离上游 agent 协议残留（Command 块 / tool_call / tool_action 标签等）。

    仅用于纯对话请求（_looks_like_agent_request 判定为 False）：coding agent
    请求需要保留模型输出的工具调用语法（下游自己解析），不能走本函数。
    """
    if not any(k in text for k in (
        "Command", "tool_call", "tool_action", "tool_name",
        "tool_callback", "<command", "arg_value", "arg_key",
        "think", "执行过程发生错误",
    )):
        return text
    for pat in _LEAK_RE:
        text = pat.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# 泄漏标签名（流式扣留判断用；大小写不敏感）
_LEAK_TAG_NAMES = ("tool_action", "tool_name", "tool_call", "tool_callback", "think", "command")


def _stream_hold_pos(buf: str, tags: tuple[str, ...] = _LEAK_TAG_NAMES) -> int:
    """返回应扣留的起始位置（-1 表示整段可安全释放）。

    两种扣留场景：
    1. 存在未闭合的标签开启（如 <tool_action> 还没等到 </tool_action>）
       ——必须等闭合标签到齐后整块处理，否则会出现"剥了外壳留下内脏"
       的残骸（下游收到的正是这种，解析器不认）
    2. 尾部的裸 "<" 是某个标签名的前缀（如 "<tool_ac"）——扣住防止
       标签名被 SSE 分包分裂
    """
    low = buf.lower()
    best = -1
    for tag in tags:
        start = 0
        opener = f"<{tag}"
        while True:
            i = low.find(opener, start)
            if i == -1:
                break
            after = low[i + len(opener): i + len(opener) + 1]
            if after in ("", ">", "/", " ", "\t", "\n"):
                closer = f"</{tag}>"
                if low.find(closer, i) == -1 and (best == -1 or i < best):
                    best = i
            start = i + 1
    if best != -1:
        return best
    j = buf.rfind("<")
    if j != -1 and ">" not in buf[j:]:
        partial = buf[j + 1:].lower().lstrip("/")
        if any(t.startswith(partial) for t in _LEAK_TAG_NAMES):
            return j
    return -1


class _StreamLeakCleaner:
    """流式泄漏清洗器：跨 chunk 缓冲 + 整块清洗。

    之前的实现按 chunk 逐段调 _sanitize_agent_leak，标签跨 SSE 分包分裂时
    整块正则匹配不上、孤儿标签正则只剥掉首尾外壳，内部内容反而漏出去。
    本清洗器把"可能不完整"的尾部扣在缓冲区，只释放确定安全的前缀。
    """

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, text: str) -> str:
        self._buf += text
        pos = _stream_hold_pos(self._buf)
        if pos == -1:
            safe, self._buf = self._buf, ""
        else:
            safe, self._buf = self._buf[:pos], self._buf[pos:]
        return _sanitize_agent_leak(safe)

    def flush(self) -> str:
        out = _sanitize_agent_leak(self._buf)
        self._buf = ""
        return out

# 工具调用流式标签（教学格式 + Trae 原生格式的标签名）
# tool_callback：deepseek-v4-flash 实测的伪回调头（开标签 + 意图文本、不闭合），
# 不扣留的话会当正文流出（下游 UI 直接看到 <tool_callback>…）；扣留后交给
# _parse_tool_calls 统一剥离
_TOOL_STREAM_TAGS = ("tool_call", "tool_calls", "tool_action", "tool_name", "tool_callback", "command", "arg_key", "arg_value")

# 裸 JSON 工具调用候选前缀：模型偶尔省略标签，直接在行首输出
# {"name": ...} 或 [{"name": ...}]（数组包多个调用）
# deepseek-v4-pro 偶尔在 { 和 " 之间加空格：{ "name": ... }
_BARE_CANDS = ('{"name"', '{ "name"', '[{"name"', '[{ "name"', '[ {"name"')
_BARE_RE = re.compile(r'(?m)^[ \t]*(\[\s*\{\s*"name"|\{\s*"name")')

# XML 属性风格工具调用：<tool_call name="..." command="..." />
# 属性值里可能含 '>'（如 shell 命令 2>/dev/null），不能用简单正则匹配整标签，
# 用起始正则定位 + 引号感知扫描
_ATTR_TAG_START_RE = re.compile(r"<(?:tool_call|tool_calls|tool_action)\b", re.I)
_ATTR_VAL_RE = re.compile(r"(\w+)\s*=\s*(?:\"([^\"]*)\"|'([^']*)')")

# 函数调用表达式参数值：k="v" / k='v' / k=7（数字）/ k=true|false|null（裸字面量）
_KV_VAL_RE = re.compile(
    r"(\w+)\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|(-?\d+(?:\.\d+)?)|(true|false|null))",
    re.I,
)

# 行首函数调用表达式（裸函数语法兜底的流式扣留锚点）
_LINE_CALL_EXPR_RE = re.compile(r"(?m)^[ \t]*([A-Za-z_][\w.]*)[ \t]*\(")

_CALL_NAME_RE = re.compile(r"([A-Za-z_][\w.]*)\s*\(")


def _parse_kv_args(body: str) -> dict[str, Any]:
    """解析函数调用表达式参数体 k=v, k2=v2 -> dict（保留键原大小写）。

    相比 _ATTR_VAL_RE 额外支持未加引号的数字/布尔/null（glm-5.3 实测输出
    web_search(query="...", max_results=7, ...)，max_results 不带引号，
    旧正则会直接丢掉这个参数）。
    """
    args: dict[str, Any] = {}
    for m in _KV_VAL_RE.finditer(body):
        key = m.group(1)
        if m.group(2) is not None:
            val: Any = m.group(2)
        elif m.group(3) is not None:
            val = m.group(3)
        elif m.group(4) is not None:
            f = float(m.group(4))
            val = int(f) if f.is_integer() else f
        else:
            val = {"true": True, "false": False, "null": None}[
                m.group(5).lower()]
        args[key] = val
    return args


def _find_call_exprs(
    text: str,
    names: frozenset[str] | set[str] | None = None,
) -> list[tuple[str, dict[str, Any], int, int]]:
    """扫描文本里的函数调用表达式 name(k=v, ...)。

    引号感知 + 括号配对扫描（参数值里可能含括号/逗号），连续多个调用
    （空格/换行分隔）逐个返回。names 传入时只接受已知工具名（裸文本
    防误伤）；None 时任意名字都收（调用块内模型已显式标记是调用）。
    返回 [(name, args_dict, start, end), ...]，end 为右括号后一位。
    """
    results: list[tuple[str, dict[str, Any], int, int]] = []
    pos = 0
    n = len(text)
    while True:
        m = _CALL_NAME_RE.search(text, pos)
        if not m:
            break
        name = m.group(1)
        # 引号感知扫描到配对的右括号
        j = m.end()
        depth = 1
        quote: str | None = None
        end = -1
        while j < n:
            ch = text[j]
            if quote is not None:
                if ch == quote:
                    quote = None
            elif ch in ('"', "'"):
                quote = ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    end = j
                    break
            j += 1
        if end == -1:
            # 括号不闭合（截断流）：跳过名字继续扫
            pos = m.end()
            continue
        if names is None or name in names:
            body = text[m.end():end]
            args = _parse_kv_args(body)
            if not args:
                args = {"input": body.strip()}
            results.append((name, args, m.start(), end + 1))
        pos = end + 1
    return results


def _colon_marker_re(tools: frozenset[str] | set[str], anchored: bool = True) -> re.Pattern:
    """glm-5.3 连写变体「工具名+参数名+冒号」的起始标记正则。

    模型省略括号/引号/等号，输出 web_searchquery: <自由文本>；工具名与
    参数名之间无空格。按工具名长度降序拼接分支避免前缀撞名。
    anchored=True 时首个标记必须在行首；连续调用时下一个标记会与
    上一个值粘连（methodsweb_searchquery:），用 anchored=False 续扫。
    命名捕获组：tool=工具名，param=参数名。
    """
    alt = "|".join(
        sorted((re.escape(t) for t in tools), key=len, reverse=True))
    body = rf"(?P<tool>(?:{alt}))(?P<param>[a-z_][a-z0-9_]*)\s*:"
    if anchored:
        return re.compile(rf"(?m)^[ \t]*{body}")
    return re.compile(body)


def _find_colon_joined_calls(
    text: str,
    tools: frozenset[str] | set[str],
) -> list[tuple[str, dict[str, Any], int, int]]:
    """解析 web_searchquery: <自由文本> 连写形态的工具调用。

    glm-5.3 实测：模型连括号都省掉，直接写
    ``web_searchquery: how to ... methodsweb_searchquery: 检测套壳...``
    （连续多个时上一个值直接拼到下一个标记前，无分隔符）。
    首个标记必须位于行首；后续标记由上一个值的结束边界界定。
    返回 [(name, args_dict, start, end), ...]，end 为值结束位置。
    """
    first = _colon_marker_re(tools).search(text)
    if not first:
        return []
    marks = [first]
    follow_re = _colon_marker_re(tools, anchored=False)
    while True:
        m2 = follow_re.search(text, marks[-1].end())
        if not m2:
            break
        marks.append(m2)
    results: list[tuple[str, dict[str, Any], int, int]] = []
    n = len(text)
    for i, mk in enumerate(marks):
        val_start = mk.end()
        val_end = marks[i + 1].start() if i + 1 < len(marks) else n
        value = text[val_start:val_end]
        # 去掉值末尾的 markdown 水平分隔线行（如模型另起一行写的 ---）
        value = re.sub(r"(?m)^[ \t]*-{3,}[ \t]*$", "", value)
        # 自由文本压成单行
        value = re.sub(r"\s+", " ", value).strip(" \t\r\n-")
        if not value:
            continue
        results.append((
            mk.group("tool"),
            {mk.group("param"): value},
            mk.start("tool"),
            val_end,
        ))
    return results


def _extract_attr_calls(rest: str, calls: list[dict[str, Any]]) -> str:
    """提取 XML 属性风格的工具调用标签，返回剩余文本。

    形如 <tool_call name="shell" command="ls" intent="..." />（自闭合或
    开标签均可）。属性值内的 '>' 不会截断标签（扫描时跳过引号内字符）。
    """
    out: list[str] = []
    pos = 0
    n = len(rest)
    while True:
        m = _ATTR_TAG_START_RE.search(rest, pos)
        if not m:
            out.append(rest[pos:])
            break
        i = m.start()
        # 扫描到真正的 '>'（跳过引号内的字符）
        j = i + 1
        quote = None
        end = -1
        while j < n:
            ch = rest[j]
            if quote is not None:
                if ch == quote:
                    quote = None
            elif ch in ('"', "'"):
                quote = ch
            elif ch == ">":
                end = j
                break
            j += 1
        if end == -1:
            out.append(rest[pos:])
            break
        tag_text = rest[i:end + 1]
        attrs: dict[str, str] = {}
        for k, v1, v2 in _ATTR_VAL_RE.findall(tag_text):
            attrs[k.lower()] = v1 if v2 == "" else v2
        name = attrs.get("name") or attrs.get("tool") or ""
        if not name:
            # 没有名字（教学格式的开标签/复数容器壳）：保留原文交给后续清理
            out.append(rest[pos:end + 1])
            pos = end + 1
            continue
        raw = attrs.get("command") or attrs.get("arguments") or attrs.get("args") or ""
        if raw.strip().startswith("{"):
            try:
                args = json.loads(raw)
            except ValueError:
                args = {"command": raw}
        else:
            args = {"command": raw}
            if attrs.get("intent"):
                args["intent"] = attrs["intent"]
        out.append(rest[pos:i])
        calls.append(_mk_tool_call(
            len(calls), name, json.dumps(args, ensure_ascii=False)))
        pos = end + 1
    return "".join(out)


def _tool_names(tools: list[dict[str, Any]] | None) -> frozenset[str] | None:
    """从 OpenAI tools 定义里提取工具名集合。"""
    if not tools:
        return None
    names = set()
    for t in tools:
        f = t.get("function") if isinstance(t, dict) else None
        if isinstance(f, dict) and f.get("name"):
            names.add(f["name"])
    return frozenset(names) or None


def _valid_bare_call_obj(obj: Any, known_tools: frozenset[str]) -> bool:
    """裸 JSON 是否是合法的工具调用对象。

    严格校验（避免误伤正文里的普通 JSON）：name 必须是已知工具名。
    两种形态放行：
    - 标准形态：带 arguments/args，且除 name/tool/intent 外无杂键
      （intent 是 ethan 教的说明字段，放行）；
    - 扁平形态（glm-5.3 实测）：name 与参数平铺——
      {"name": "web_search", "query": "...", "max_results": 10}，
      无 arguments 包裹层，但至少带一个参数键。
    """
    if not isinstance(obj, dict):
        return False
    name = obj.get("name")
    if not isinstance(name, str) or name not in known_tools:
        return False
    if "arguments" in obj or "args" in obj:
        return set(obj) <= {"name", "arguments", "args", "intent", "tool"}
    # 扁平形态：name/tool 之外还有键即视为参数平铺
    return bool(set(obj) - {"name", "tool"})


def _flatten_call_args(obj: dict[str, Any]) -> dict[str, Any]:
    """从裸 JSON 调用对象提取参数；扁平形态（无 arguments 层）取剩余键。

    glm-5.3 还会把 arguments 输出成 JSON 字符串（而非对象）：
    {"name": "web_search", "arguments": "{\"query\": \"...\", \"max_results\": 10}"}
    —— 解包成 dict，避免下游拿到 {"input": "<整段 JSON 字符串>"}。
    """
    args = obj.get("arguments", obj.get("args"))
    if args is None:
        args = {k: v for k, v in obj.items() if k not in ("name", "tool")}
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            if isinstance(parsed, dict):
                args = parsed
        except Exception:
            pass
    return args if isinstance(args, dict) else {"input": args}


def _quote_is_close(s: str, i: int) -> bool:
    """Heuristic: does the ``"`` at position *i* close the current string?

    Tolerates unescaped quotes inside values (deepseek v4 pro leak): a quote
    followed by ``}`` / ``]`` only closes when valid structure tokens follow
    the bracket — a ``}`` immediately followed by more string content
    (e.g. ``echo "}" && ls``) is an inner quote, not the end of the value.
    """
    n = len(s)
    j = i + 1
    while j < n and s[j] in " \t":
        j += 1
    if j >= n:
        return True
    c = s[j]
    if c in ":,\n":
        return True
    if c in "}]":
        k = j + 1
        while k < n and s[k] in " \t\r\n":
            k += 1
        if k >= n or s[k] in ",}]\n":
            return True
        return False
    return False


def _find_obj_extent(text: str, start: int) -> int:
    """Best-effort brace-depth scan to find the end of a JSON object.

    Returns the index *after* the matching ``}`` or -1 if unbalanced.
    Tracks string state with :func:`_quote_is_close` so stray braces inside
    broken string values (e.g. ``"command": "echo "}" && ls"``) don't
    prematurely zero the depth.  If the boundary still can't be found the
    caller degrades gracefully (no repair, raw text preserved).
    """
    if start >= len(text) or text[start] != "{":
        return -1
    depth = 0
    in_string = False
    i = start
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\\" and i + 1 < n:
            i += 2
            continue
        if ch == '"':
            if not in_string:
                in_string = True
            elif _quote_is_close(text, i):
                in_string = False
            i += 1
            continue
        if not in_string:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i + 1
        i += 1
    return -1


def _repair_json_quotes(s: str) -> str | None:
    """Try to fix unescaped double-quotes inside JSON string values.

    DeepSeek v4 pro occasionally emits shell commands with unescaped ``"``
    (e.g. ``echo "---"``), which makes the JSON invalid.  Walk through the
    text tracking string-state; quotes that don't look like a real string
    close (per :func:`_quote_is_close`) are escaped in place.

    Returns the repaired string, or *None* if no repair was applied or the
    input doesn't look like a JSON object/array.
    """
    stripped = s.lstrip()
    if not stripped or stripped[0] not in "{[":
        return None
    out: list[str] = []
    i = 0
    n = len(s)
    in_string = False
    repaired_any = False
    while i < n:
        ch = s[i]
        if not in_string:
            out.append(ch)
            if ch == '"':
                in_string = True
            i += 1
            continue
        if ch == "\\" and i + 1 < n:
            out.append(s[i : i + 2])
            i += 2
            continue
        if ch == '"':
            if _quote_is_close(s, i):
                out.append(ch)
                in_string = False
            else:
                out.append('\\"')
                repaired_any = True
            i += 1
            continue
        out.append(ch)
        i += 1
    if not repaired_any:
        return None
    return "".join(out)


class _StreamToolCallSplitter:
    """流式工具调用切分器：正文按 chunk 透传，工具调用块整段扣留。

    调用块（无论是否已闭合）都要在 flush 时统一解析成 OpenAI tool_calls，
    不能当正文放行——所以从首个疑似标签起全部扣留（与泄漏清洗器的
    "闭合即放行"语义不同）。

    known_tools：请求携带的工具名集合。传入后额外扣留「行首裸 JSON」候选
    （模型偶尔省略标签、直接输出 {"name": ...} 裸调用，实测 deepseek-v4-pro
    连续多轮调用后会出现这种偷懒写法）。
    """

    def __init__(self, known_tools: frozenset[str] | None = None) -> None:
        self._buf = ""
        self._tools = frozenset(known_tools) if known_tools else None

    @staticmethod
    def _first_tag_pos(buf: str) -> int:
        low = buf.lower()
        best = -1
        for tag in _TOOL_STREAM_TAGS:
            start = 0
            opener = "<" + tag
            while True:
                i = low.find(opener, start)
                if i == -1:
                    break
                after = low[i + len(opener): i + len(opener) + 1]
                if after in ("", ">", "/", " ", "\t", "\n"):
                    if best == -1 or i < best:
                        best = i
                    break
                start = i + 1
        if best != -1:
            return best
        # 尾部裸 "<" 可能是分裂的标签名前缀，扣住防漏
        j = buf.rfind("<")
        if j != -1 and ">" not in buf[j:]:
            partial = buf[j + 1:].lower().lstrip("/")
            if any(t.startswith(partial) for t in _TOOL_STREAM_TAGS):
                return j
        return -1

    def _bare_pos(self, buf: str) -> int:
        """行首裸 JSON 调用候选位置（含跨 chunk 分裂的前缀）。"""
        if not self._tools:
            return -1
        best = -1
        m = _BARE_RE.search(buf)
        if m:
            best = m.start(1)
        # 尾部行可能是分裂中的候选前缀（如 buf 以 '{"na' 结尾）
        last_nl = buf.rfind("\n")
        line = buf[last_nl + 1:]
        ls = line.lstrip()
        if ls and any(c.startswith(ls) and len(ls) < len(c) for c in _BARE_CANDS):
            p = last_nl + 1 + (len(line) - len(ls))
            if best == -1 or p < best:
                best = p
        return best

    def _call_expr_pos(self, buf: str) -> int:
        """行首已知工具名调用表达式的扣留位置（glm-5.3 裸函数语法变体）。

        模型偶尔连 <tool_call> 标签都省掉，直接在行首输出
        web_search(query="...", ...)（可连续多个，尾随孤立 </arg_value>）。
        行首 + 已知工具名 + 紧跟 "(" 的组合在正文里极其罕见，值得扣留。
        含跨 chunk 分裂的前缀（尾行是纯工具名前缀、可能 ( 在下一个 chunk）。
        """
        if not self._tools:
            return -1
        best = -1
        for m in _LINE_CALL_EXPR_RE.finditer(buf):
            if m.group(1) in self._tools:
                if best == -1 or m.start() < best:
                    best = m.start()
        # 尾部行可能是分裂中的候选前缀（如 buf 以行首 'web_sear' 结尾）
        last_nl = buf.rfind("\n")
        line = buf[last_nl + 1:]
        ls = line.lstrip()
        if ls and re.fullmatch(r"[A-Za-z_][\w.]*", ls) and len(ls) <= max(
                len(t) for t in self._tools):
            if any(t.startswith(ls) for t in self._tools):
                p = last_nl + 1 + (len(line) - len(ls))
                if best == -1 or p < best:
                    best = p
        return best

    def _colon_pos(self, buf: str) -> int:
        """行首冒号连写调用的扣留位置（glm-5.3 web_searchquery: 变体）。

        完整标记 web_searchquery: 直接正则定位；跨 chunk 分裂的前缀
        （web_searchqu / web_searchquery / web_searchquery:）用尾行
        前缀匹配扣住。
        """
        if not self._tools:
            return -1
        if not hasattr(self, "_colon_re"):
            self._colon_re = _colon_marker_re(self._tools)
        best = -1
        m = self._colon_re.search(buf)
        if m:
            best = m.start("tool")
        last_nl = buf.rfind("\n")
        line = buf[last_nl + 1:]
        ls = line.lstrip()
        if ls:
            for t in self._tools:
                if ls.startswith(t):
                    tail = ls[len(t):]
                    if tail and re.fullmatch(r"[a-z0-9_]*:?", tail):
                        p = last_nl + 1 + (len(line) - len(ls))
                        if best == -1 or p < best:
                            best = p
                        break
        return best

    def feed(self, text: str) -> str:
        self._buf += text
        pos = self._first_tag_pos(self._buf)
        if self._tools:
            bp = self._bare_pos(self._buf)
            if bp != -1 and (pos == -1 or bp < pos):
                pos = bp
            ep = self._call_expr_pos(self._buf)
            if ep != -1 and (pos == -1 or ep < pos):
                pos = ep
            cp = self._colon_pos(self._buf)
            if cp != -1 and (pos == -1 or cp < pos):
                pos = cp
        if pos == -1:
            safe, self._buf = self._buf, ""
        else:
            safe, self._buf = self._buf[:pos], self._buf[pos:]
        return safe

    def flush(self) -> tuple[str, list[dict[str, Any]]]:
        rest, calls = _parse_tool_calls(self._buf, self._tools)
        self._buf = ""
        return rest, calls


def _looks_like_agent_request(messages: list[dict[str, Any]], body: dict[str, Any]) -> bool:
    """判断请求是否来自自带工具协议的下游（coding agent / function calling）。

    这类请求里模型输出的工具调用语法（<tool_action> 等）是下游 agent 自己
    要解析的，不能注入压制指令、也不能清洗（实测模型会完美遵循下游 system
    prompt 里教的格式；反而我们的 guard 注入会让模型输出两段指令打架的
    纠结文本，逐 chunk 清洗会把 tool_action 外壳剥掉留下内脏残骸）。

    判定信号（任一命中即视为 agent 请求）：
    1. 请求带 tools / tool_choice 字段（标准 function calling）
    2. messages 里有 role=tool/function 的消息（工具结果回传）
    3. system 消息里有工具语法教学（<tool_action / <tool_name / tool_call /
       execute_command 等）
    4. 历史消息正文里出现过完整的工具调用块（agent 循环中的多轮请求）
    """
    if body.get("tools") or body.get("tool_choice") is not None:
        return True
    for m in messages:
        role = m.get("role", "")
        if role in ("tool", "function"):
            return True
        content = m.get("content", "")
        if isinstance(content, list):
            content = "".join(
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        if not isinstance(content, str) or not content:
            continue
        low = content.lower()
        if role == "system":
            if any(k in low for k in (
                "<tool_action", "<tool_name", "tool_call", "<function_call",
                "execute_command", "tool use", "工具调用", "你可以调用工具",
            )):
                return True
        else:
            # 非系统消息只认强信号：出现过完整的工具调用块
            if "<tool_action" in low and "</tool_action>" in low:
                return True
    return False


# prompt-based function calling 的教学格式标签。选 XML 标签 + JSON 参数：
# 标签与模型原生 SOLO 协议同构（遵循度最高），JSON 参数适配任意工具 schema
_TC_OPEN = '<tool_call>'
_TC_CLOSE = '</tool_call>'
_TOOL_DECODER = json.JSONDecoder()


def _build_tools_system(tools: list[dict[str, Any]]) -> str:
    """把 OpenAI tools 定义序列化成 prompt-based function calling 教学指令。

    Trae 上游不支持原生 tools 参数，用提示词教学 + 输出解析的方式模拟。
    """
    defs = []
    for t in tools:
        f = t.get("function") if isinstance(t, dict) else None
        if not isinstance(f, dict) or not f.get("name"):
            continue
        defs.append({
            "name": f.get("name", ""),
            "description": f.get("description", ""),
            "parameters": f.get("parameters", {}),
        })
    if not defs:
        return ""
    return (
        "你可以调用下列工具（function calling）。工具列表：\n"
        + json.dumps(defs, ensure_ascii=False)
        + "\n\n调用规则：\n"
        "- 需要调用工具时，输出如下格式的调用块（JSON 一行，可连续多个块）：\n"
        + _TC_OPEN + '\n{"name": "工具名", "arguments": {参数对象}}\n' + _TC_CLOSE + "\n"
        "- arguments 必须是合法 JSON 对象，与工具参数 schema 一致。\n"
        "- 每次调用都必须带完整的开闭标签，即使是同一任务里的第 N 次调用；"
        "禁止省略标签直接输出裸 JSON。\n"
        "- 调用块外可以写简短的说明文字；不要把工具参数写进正文；"
        "不要发明列表外的工具。\n"
        "- 收到 [tool_result] 开头的消息后，那是工具执行结果，据此继续任务，"
        "直到可以给出最终回答。如果对话里还没有出现 [tool_result]，"
        "说明工具从未执行过——不要假设命令已运行、不要等待结果，"
        "需要结果就直接再输出调用块。\n"
    )


def _serialize_tool_calls(tool_calls: list[dict[str, Any]]) -> str:
    """把历史 assistant.tool_calls 序列化成教学格式的调用块文本。"""
    parts = []
    for c in tool_calls:
        f = c.get("function") or {}
        try:
            args = json.loads(f.get("arguments") or "{}")
        except Exception:
            args = {"raw": f.get("arguments", "")}
        parts.append(_TC_OPEN + "\n" + json.dumps(
            {"name": f.get("name", ""), "arguments": args}, ensure_ascii=False
        ) + "\n" + _TC_CLOSE)
    return "\n".join(parts)


def _mk_tool_call(idx: int, name: str, arguments: str) -> dict[str, Any]:
    return {
        "id": f"call_{idx:02d}_{uuid.uuid4().hex[:20]}",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _parse_tool_calls(
    content: str,
    known_tools: frozenset[str] | set[str] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """从模型输出里解析工具调用块，返回 (剩余正文, tool_calls 列表)。

    主路径解析教学格式（JSON 参数，用 raw_decode 正确处理嵌套大括号）；
    兜底一：Trae 原生 SOLO XML（<tool_name>/<command>，名字对不上由
    下游 agent 报错纠偏）；
    兜底二：无标签裸 JSON（known_tools 提供时启用——模型连续多轮调用后
    会偷懒省略标签，直接在行首输出 {"name": ...}）。
    """
    # 容错归一化：模型偶尔把标签写成复数变体（开/闭合都可能，大小写不定），
    # 实测出现过「标准单数开标签 + 复数闭标签」的混搭——严格匹配会解析失败，
    # 兜底清理只剥开标签、留下复数闭标签和裸 JSON 整段泄漏进正文。
    # 先统一归一成教学的标准单数标签再进主解析。
    content = re.sub(r"</\s*tool_calls\s*>", _TC_CLOSE, content, flags=re.I)
    content = re.sub(r"<\s*tool_calls\s*>", _TC_OPEN, content, flags=re.I)
    # 伪回调头 <tool_callback>（deepseek-v4-flash 实测）必须在主解析前剥离：
    # 闭合块整块剥；未闭合剥到下一个 <tool_call> 为止——主循环会把
    # <tool_call> 消费掉，放后面做前瞻就没有锚点了。剥法与 _LEAK_RE 统一。
    content = re.sub(r"<tool_callback[^>]*>.*?</tool_callback>", "", content, flags=re.S)
    content = re.sub(r"<tool_callback[^>]*>.*?(?=<tool_call[>\s])", "", content, flags=re.S)
    # 记录原文是否带 arg_key/arg_value 残骸：glm-5.3 的坏调用块伴随孤立
    # </arg_value>，这是「这段文本确实是坏掉的调用」而非正文代码示例的强信号
    had_arg_debris = bool(re.search(r"</?\s*arg_(?:key|value)", content))
    calls: list[dict[str, Any]] = []
    rest_parts: list[str] = []
    pos = 0
    n = len(content)
    while pos < n:
        i = content.find(_TC_OPEN, pos)
        if i == -1:
            rest_parts.append(content[pos:])
            break
        rest_parts.append(content[pos:i])
        j = i + len(_TC_OPEN)
        block_calls: list[tuple[str, str]] = []
        closed = False
        expr_consumed = False  # 块内表达式路径已整块消费
        # 块内循环：连续解码多个 JSON（复数容器装多个调用），直到闭标签
        while True:
            while j < n and content[j] in " \t\r\n":
                j += 1
            if j >= n:
                break
            if content.startswith(_TC_CLOSE, j):
                j += len(_TC_CLOSE)
                closed = True
                break
            if content[j] == "{":
                try:
                    obj, end = _TOOL_DECODER.raw_decode(content, j)
                except ValueError:
                    raw_end = _find_obj_extent(content, j)
                    if raw_end != -1:
                        repaired = _repair_json_quotes(content[j:raw_end])
                        if repaired:
                            try:
                                obj = json.loads(repaired)
                            except (ValueError, TypeError):
                                obj = None
                            if isinstance(obj, dict):
                                nm = obj.get("name") or obj.get("tool") or ""
                                ag = obj.get("arguments", obj.get("args", {}))
                                if not isinstance(ag, dict):
                                    ag = {"input": ag}
                                if nm:
                                    block_calls.append(
                                        (str(nm), json.dumps(ag, ensure_ascii=False)))
                                j = raw_end
                                continue
                    break
                name = obj.get("name") or obj.get("tool") or ""
                args = obj.get("arguments", obj.get("args", {}))
                if not isinstance(args, dict):
                    args = {"input": args}
                if name:
                    block_calls.append(
                        (str(name), json.dumps(args, ensure_ascii=False)))
                j = end
                continue
            # 不是 JSON：可能是 glm-5.3 的函数调用表达式写法——
            # <tool_call>web_search(query="...", max_results=7) web_search(...)</arg_value></tool_call>
            # （块内可连续多个调用表达式，混孤立 </arg_value> 残骸）。
            # 引号感知扫到块尾，全部转成 tool_calls 后整块消费。
            close = content.find(_TC_CLOSE, j)
            block_text = content[j:close if close != -1 else n]
            exprs = _find_call_exprs(block_text, None)
            if exprs:
                for name, args, _s, _e in exprs:
                    calls.append(_mk_tool_call(
                        len(calls), name, json.dumps(args, ensure_ascii=False)))
                pos = (close + len(_TC_CLOSE)) if close != -1 else n
                expr_consumed = True
                break
            break  # 既不是 JSON 也不是闭标签：块不合法
        if closed and block_calls:
            for name, args_json in block_calls:
                calls.append(_mk_tool_call(len(calls), name, args_json))
            pos = j
            continue
        if expr_consumed:
            continue
        # 不是合法调用块：保留原文继续扫描
        rest_parts.append(content[i:i + len(_TC_OPEN)])
        pos = i + len(_TC_OPEN)
    rest = "".join(rest_parts)

    # 残余噪音剥离（无论是否已解析出调用都要做）：
    # - 孤立 tool_callback 标签壳：未闭合到流尾的（模型放弃调用继续说正文）
    #   只剥壳、保留内部文本，不能整段吞掉
    # - 孤立 arg_key / arg_value 标签：glm-5.3 函数调用语法残骸
    rest = re.sub(r"</?\s*tool_callback[^>]*>", "", rest)
    rest = re.sub(r"</?\s*arg_(?:key|value)[^>]*>", "", rest)
    rest = re.sub(r"\n{3,}", "\n\n", rest)

    # 兜底：Trae 原生 <tool_name>/<command>（仅当教学格式没解析出任何调用时）
    if not calls:
        # 兜底三：函数调用语法（glm-5.3 实测：无视 JSON 教学，标签内直接写
        # skill_read(intent="...")，常混入孤立 </arg_value> 残骸）
        def _grab_fn_call(match: re.Match) -> str:
            name, body = match.group(1), match.group(2)
            args = _parse_kv_args(body)
            if not args:
                args = {"input": body.strip()}
            calls.append(_mk_tool_call(
                len(calls), name.strip(), json.dumps(args, ensure_ascii=False)))
            return ""

        # ) 与 </tool_call> 之间允许夹孤立残骸标签（</arg_value> 等）
        fn_call_re = re.compile(
            r"<tool_call>\s*([A-Za-z_][\w.]*)\s*\(([\s\S]*?)\)\s*"
            r"(?:</[^>]+>\s*)*</tool_call>",
            re.I,
        )
        rest = fn_call_re.sub(_grab_fn_call, rest)
        if calls:
            return rest.strip(), calls

        def _grab_native(match: re.Match) -> str:
            calls.append(_mk_tool_call(len(calls), match.group(1), json.dumps(
                {"command": match.group(2).strip()}, ensure_ascii=False)))
            return ""

        native_re = re.compile(
            r"<tool_name>\s*([^<\s]+)\s*</tool_name>\s*<command>\s*([\s\S]*?)\s*</command>",
            re.S,
        )
        rest = native_re.sub(_grab_native, rest)

        # 兜底：XML 属性风格 <tool_call name="..." command="..." />
        # （实测 deepseek-v4-pro 偶发输出这种自闭合属性标签）
        rest = _extract_attr_calls(rest, calls)
        rest = re.sub(r"</?tool_action[^>]*>", "", rest)
        rest = re.sub(re.escape(_TC_OPEN) + r"\s*", "", rest)
        rest = re.sub(r"\s*" + re.escape(_TC_CLOSE), "", rest)

    # 兜底二：无标签裸 JSON（仅 agent 请求、且前两种格式都没解析出调用）
    if not calls and known_tools:
        known = frozenset(known_tools)
        # 同一行连续多个裸 JSON（deepseek-v4-pro 实测）：在 } {"name" 边界
        # 插入换行，使后续对象也能被 _BARE_RE 的行首锚点匹配到
        rest = re.sub(r'}\s+(?=\{\s*"name")', '}\n', rest)
        rest_parts2: list[str] = []
        pos = 0
        while True:
            m = _BARE_RE.search(rest, pos)
            if not m:
                rest_parts2.append(rest[pos:])
                break
            js = m.start(1)
            try:
                obj, end = _TOOL_DECODER.raw_decode(rest, js)
            except ValueError:
                raw_end = _find_obj_extent(rest, js)
                if raw_end != -1:
                    repaired = _repair_json_quotes(rest[js:raw_end])
                    if repaired:
                        try:
                            obj = json.loads(repaired)
                        except (ValueError, TypeError):
                            obj = None
                        if obj:
                            items = obj if isinstance(obj, list) else [obj]
                            if all(_valid_bare_call_obj(it, known) for it in items):
                                rest_parts2.append(rest[pos:m.start()].rstrip())
                                for it in items:
                                    nm = it.get("name") or it.get("tool") or ""
                                    ag = _flatten_call_args(it)
                                    calls.append(_mk_tool_call(
                                        len(calls), str(nm),
                                        json.dumps(ag, ensure_ascii=False)))
                                pos = raw_end
                                continue
                rest_parts2.append(rest[pos:m.end()])
                pos = m.end()
                continue
            items = obj if isinstance(obj, list) else [obj]
            if obj and all(_valid_bare_call_obj(it, known) for it in items):
                rest_parts2.append(rest[pos:m.start()].rstrip())
                for it in items:
                    name = it.get("name") or it.get("tool") or ""
                    args = _flatten_call_args(it)
                    calls.append(_mk_tool_call(
                        len(calls), str(name), json.dumps(args, ensure_ascii=False)))
                pos = end
            else:
                rest_parts2.append(rest[pos:m.end()])
                pos = m.end()
        rest = "".join(rest_parts2)

    # 兜底四：裸函数调用表达式（glm-5.3 实测 Scopus 会话变体：
    # web_search(query="...", max_results=7) web_search(...)</arg_value>
    # ——无 <tool_call> 包裹、连续多个调用 + 孤立 </arg_value> 残骸）。
    # 门槛（防误伤正文里的普通代码示例）：
    # a) 前面所有格式都没解出调用；b) 工具名在请求的 known_tools 里；
    # c) 原文带 arg_* 残骸（坏调用块的强信号），或表达式位于行首
    # （流式 splitter 对行首已知工具名调用会主动扣留，两端约定一致）。
    if not calls and known_tools and (had_arg_debris or _LINE_CALL_EXPR_RE.search(rest)):
        known = frozenset(known_tools)
        exprs = _find_call_exprs(rest, known)
        # 有残骸信号时全部收；没有残骸时只收行首的（inline 的可能是正文示例）
        if not had_arg_debris:
            exprs = [e for e in exprs
                     if e[2] == 0 or rest[e[2] - 1] == "\n"]
        if exprs:
            out: list[str] = []
            pos3 = 0
            for name, args, s, e in exprs:
                out.append(rest[pos3:s])
                calls.append(_mk_tool_call(
                    len(calls), name, json.dumps(args, ensure_ascii=False)))
                pos3 = e
            out.append(rest[pos3:])
            rest = "".join(out)

    # 兜底五：冒号连写形态（glm-5.3 实测：web_searchquery: <自由文本>，
    # 模型省略括号/引号/等号；连续多个时上一个值直接拼到下一个标记前）。
    # 门槛与裸表达式一致：仅在已知工具名、且前面格式都没解出调用时启用；
    # 首个标记必须在行首（splitter 端会主动扣留这类行首前缀）。
    if not calls and known_tools:
        colon_calls = _find_colon_joined_calls(rest, frozenset(known_tools))
        if colon_calls:
            out: list[str] = []
            pos5 = 0
            for name, args, s, e in colon_calls:
                out.append(rest[pos5:s])
                calls.append(_mk_tool_call(
                    len(calls), name, json.dumps(args, ensure_ascii=False)))
                pos5 = e
            out.append(rest[pos5:])
            rest = "".join(out)
            rest = re.sub(r"(?m)^[ \t]*-{3,}[ \t]*\n?", "", rest)

    # glm-5.3 实测：正文带 <think>...</think>\n\n</think>（多一个游离闭标签）。
    # agent 模式不走 _sanitize_agent_leak，think 清洗在这里兜住。
    if "<think>" in rest or "</think>" in rest:
        rest = re.sub(r"<think>.*?</think>", "", rest, flags=re.S)
        rest = re.sub(r"</?\s*think>", "", rest)
        rest = rest.strip()

    return rest.strip(), calls


def _extract_prompt(
    messages: list[dict[str, Any]],
    guard: bool = True,
    tools: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """OpenAI messages -> Trae messages（content 转成 block 数组）。

    纯对话请求（guard=True）注入 agent 协议压制指令：Trae 的 llm_utils_chat
    端点（solo_work_lite/chat_v3 等 function）服务端会给模型注入"你是带 shell
    工具的 coding agent"的预设，纯对话场景下模型会把 <Command>...</Command>
    之类工具调用语法当正文吐出来（下游 OpenAI 协议客户端不认识）。

    coding agent 请求（guard=False，由 _looks_like_agent_request 判定）不注入：
    下游 system prompt 自己教的工具格式模型会完美遵循，注入反而制造指令冲突。

    带 tools 的请求（prompt-based function calling shim）：
    - 工具教学指令追加到首条 system（没有则新建）
    - 历史 assistant.tool_calls 序列化成调用块文本（上游是纯 chat 模型，
      不认 tool_calls 字段，丢了模型就不知道自己之前调过什么）
    - role=tool 的结果消息转成 user 角色 + [tool_result] 前缀
    """
    out: list[dict[str, Any]] = []
    guard_merged = False
    tools_taught = False
    teaching = _build_tools_system(tools) if tools else ""

    def _content_parts(content: Any) -> list[dict[str, Any]]:
        """OpenAI content -> Trae block 数组，保留 image_url 等多模态 block。

        上游 llm_utils_chat 认 OpenAI 风格的 image_url block（data URL 实测
        glm-5.3-flash 能读图），此前把 content 拍平成纯文本导致图片被静默
        丢弃、模型回答"没有看到图片"。
        """
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        parts: list[dict[str, Any]] = []
        if isinstance(content, list):
            for p in content:
                if not isinstance(p, dict):
                    continue
                ptype = p.get("type")
                if ptype == "text":
                    if p.get("text"):
                        parts.append({"type": "text", "text": p.get("text", "")})
                elif ptype == "image_url":
                    iu = p.get("image_url")
                    url = iu.get("url") if isinstance(iu, dict) else iu
                    if url:
                        parts.append({"type": "image_url", "image_url": {"url": url}})
        return parts or [{"type": "text", "text": ""}]

    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")

        if role == "system" and not guard_merged:
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = "".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            else:
                text = ""
            merged = text
            if guard:
                merged = _AGENT_GUARD + "\n\n" + merged
            if teaching and not tools_taught:
                merged = merged + "\n\n" + teaching
                tools_taught = True
            out.append({"role": "system", "content": [{"type": "text", "text": merged}]})
            guard_merged = True
            continue

        parts = _content_parts(content)
        text = "".join(p["text"] for p in parts if p["type"] == "text")

        tool_calls = m.get("tool_calls") if isinstance(m, dict) else None
        if role == "assistant" and tool_calls:
            # 上游不认 tool_calls 字段：序列化成教学格式的调用块拼进正文
            block = _serialize_tool_calls(tool_calls)
            parts.append({"type": "text", "text": ("\n" + block) if text else block})
        elif role in ("tool", "function"):
            # 工具结果消息：转 user 角色 + [tool_result] 前缀（上游无此角色概念）
            name = m.get("name") or (
                (m.get("tool_call_id") and "") or "tool"
            )
            parts.insert(0, {"type": "text", "text": f"[tool_result | {name}]\n"})
            role = "user"

        out.append({"role": role, "content": parts})

    if guard and not guard_merged:
        out.insert(0, {
            "role": "system",
            "content": [{"type": "text", "text": _AGENT_GUARD}],
        })
    if teaching and not tools_taught:
        out.insert(0, {
            "role": "system",
            "content": [{"type": "text", "text": teaching}],
        })
    return out


def _build_chat_body(messages: list[dict[str, Any]], model: str, stream: bool) -> dict[str, Any]:
    session_id = str(uuid.uuid4())
    return {
        "messages": messages,
        "model": model,
        "config_name": model,  # Work 通道必须 config_name + model 成对（traework2api 实测）
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
    token = work["access_token"]
    uid = work.get("uid") or ""

    trae_model = _map_model(model)
    body = _build_chat_body(messages, trae_model, stream)
    # 绝大多数模型走 solo_work_lite；少数（glm-5.1 / Doubao-Seed-Code 等）
    # 在该 function 下 4001，需改走 chat_v3（实测）
    body["function"] = _WORK_FUNCTION_OVERRIDE.get(trae_model, "solo_work_lite")

    machine_id = work.get("machine_id") or uuid.uuid4().hex
    device_id = work.get("device_id") or hashlib.sha256(machine_id.encode()).hexdigest()[:32]
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if stream else "application/json",
        "User-Agent": f"Trae/{_TRAE_APP_VERSION}",
        "Authorization": f"Cloud-IDE-JWT {token}",
        "X-Cloudide-Token": token,
        "X-Ide-Token": token,
        "X-Uid": uid,
        "X-App-Id": X_APP_ID,
        "X-App-Version": "default",
        "X-Ide-Version": _TRAE_APP_VERSION,
        "X-Ide-Version-Code": _TRAE_APP_VERSION_CODE,
        "X-App-Version-Code": _TRAE_APP_VERSION_CODE,
        "X-Ide-Version-Type": "stable",
        "X-Device-Type": "windows",
        "X-OS-Version": "Windows 11 Pro",
        "X-Device-Brand": "83DG",
        "Request-Traffic-Type": "prod",
        "X-Machine-Id": machine_id,
        "X-Device-Id": device_id,
    }

    url = f"{BASE_URL_CN}/api/agent/v3/llm_utils_chat"
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
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


# ───────────────────────── 认证 ─────────────────────────

# 全局认证缓存（惰性加载）
_auth_cache: tuple[str, str] | None = None
# 凭证路径可用 TRAE_WORK_CRED_PATH 覆盖（默认寄存在 ~/.ethan，兼容 ethan 生态）
_WORK_CRED_PATH = Path(os.environ.get("TRAE_WORK_CRED_PATH", str(Path.home() / ".ethan" / "trae_work.json")))


def _load_work_cred() -> dict[str, Any] | None:
    """从 ~/.ethan/trae_work.json 读 Work 凭证（trae_work_login.py 生成）。"""
    if not _WORK_CRED_PATH.exists():
        return None
    try:
        return json.loads(_WORK_CRED_PATH.read_text("utf-8"))
    except Exception as e:
        log.warning("Trae Work 凭证解析失败: %s", e)
        return None


def _auth() -> tuple[str, str]:
    """获取 (token, user_id)：优先 Work 凭证，其次环境变量，最后解密 storage.json。"""
    global _auth_cache
    if _auth_cache:
        return _auth_cache

    # 0) Trae Work 凭证（trae_work_login.py 生成，独立账号体系）
    work = _load_work_cred()
    if work and work.get("access_token"):
        _auth_cache = (work["access_token"], work.get("uid") or "")
        log.info("Trae auth loaded via Work credentials (uid=%s)", work.get("uid"))
        return _auth_cache

    # 1) 环境变量（TRAE_TOKEN / TRAE_USER_ID）
    token = os.environ.get("TRAE_TOKEN", "")
    user_id = os.environ.get("TRAE_USER_ID", "")
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

    raise HTTPException(status_code=401, detail="trae not authenticated - 请先运行 trae_work_login.py 或登录 Trae IDE")


# ───────────────────────── SSE 解析 ─────────────────────────

# Trae 错误码 -> 友好中文文案（官方 docs.trae.ai/ide/error-codes）
_TRAE_ERROR_HINTS: dict[int, str] = {
    1005: "Trae plan 权益不足（当前账号/模型组合无权限，可检查会员套餐或换模型）",
    3003: "Trae 模型暂不可用（今日额度可能已用完，或当前模型无权限）",
    3004: "Trae 当前模型访问量过大，请稍后重试",
    4001: "Trae 服务端错误，请稍后重试",
    4007: "Trae 请求限流，请稍后重试",
    4010: "Trae 检测到风险账号，已自动登出，请重新登录",
    4011: "Trae AI 问答今日用量已达上限，请明日再试",
    4013: "Trae AI 服务在当前地区不可用",
    4015: "Trae 检测到账号/IP 风险，请求已被阻止",
    4021: "Trae 今日会话次数已达上限，请明日再试",
    4022: "Trae 请求失败，请尝试新建会话并重试",
    4023: "Trae 模型列表已更新，请确认后重试",
    4031: "Trae 今日请求额度已用完，请明日再试（每日 0 点重置）",
    4050: "Trae 请求超时，模型服务资源紧张，请稍后重试",
    4051: "Trae 请求超时，模型服务资源紧张，请稍后重试",
}


def _trae_error_text(data: dict[str, Any]) -> str:
    """把 Trae 错误事件转成友好中文文案（附错误码，便于定位）。"""
    code = data.get("code")
    msg = data.get("message") or data.get("error") or ""
    if isinstance(code, int) and code in _TRAE_ERROR_HINTS:
        hint = _TRAE_ERROR_HINTS[code]
        text = hint + (f"（{msg}）" if msg and msg not in hint else "")
    elif msg:
        text = f"Trae 错误: {msg}"
    else:
        text = "Trae 未知错误"
    if code is not None and str(code) not in text:
        text = f"{text} (code: {code})"
    return text

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

def _anthropic_sse(event_name: str, payload: dict[str, Any]) -> str:
    """构造一条 Anthropic SSE 事件（event + data）。"""
    return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _wrap_anthropic_stream(
    openai_stream: Iterator[str], model: str
) -> Iterator[str]:
    """把 TraeProvider._stream 的 OpenAI chat chunk SSE 流包成 Anthropic 事件流。

    /v1/messages（Claude Code 等 Anthropic 客户端）经 anthropic_to_chat 转成
    chat 请求后路由到 Trae；本包装器把 OpenAI chunk 喂给
    AnthropicStreamConverter，输出 message_start / content_block_delta /
    message_stop 等标准事件（含 reasoning_content → thinking 块、
    tool_calls → tool_use 块）。_stream 是同步生成器，此处保持同步迭代。
    """
    converter = AnthropicStreamConverter(model)
    for piece in openai_stream:
        for line in piece.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                continue
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if isinstance(chunk, dict) and chunk.get("error"):
                # 错误 chunk → Anthropic error 事件（而非塞进正文文本）
                err = chunk["error"]
                msg = str(err.get("message", err))
                code = err.get("code")
                if code is not None and str(code) not in msg:
                    msg = f"{msg} (code: {code})"
                yield _anthropic_sse("error", {
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": msg,
                    },
                })
                return
            for event_name, payload in converter.feed_chunk(chunk):
                yield _anthropic_sse(event_name, payload)
    for event_name, payload in converter.finish():
        yield _anthropic_sse(event_name, payload)
    # 与 CodeBuddy 通道的 anthropic 流保持一致：message_stop 后补 [DONE]
    # 确保 SSE 客户端立即结束等待
    yield "data: [DONE]\n\n"


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
        # coding agent 请求：不注入 guard、不清洗——下游自己解析工具调用语法
        agent_mode = _looks_like_agent_request(messages, body)
        _debug_dump(
            "debug_trae_route",
            model=requested_model,
            agent_mode=agent_mode,
            tools_count=len(body.get("tools") or []),
            stream=stream,
            message_count=len(messages),
        )

        tools = body.get("tools") or []
        prompt = _extract_prompt(
            messages, guard=not agent_mode, tools=tools if agent_mode else None
        )
        if not prompt:
            raise HTTPException(status_code=400, detail="no text content")

        if stream:
            include_usage = bool(
                (body.get("stream_options") or {}).get("include_usage"))
            # anthropic 协议没有 stream_options 字段，但 message_delta 需要
            # usage（Claude Code 靠它统计 token），这里强制向 _stream 索取
            if protocol == "anthropic":
                include_usage = True
            gen = self._stream(prompt, requested_model,
                               sanitize=not agent_mode, tools=tools or None,
                               include_usage=include_usage)
            if protocol == "anthropic":
                gen = _wrap_anthropic_stream(gen, requested_model)
            return StreamingResponse(
                gen,
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "close"},
            )
        # 非流式聚合内部是同步 urllib 调用（最长 180s），放线程池执行，
        # 避免阻塞事件循环拖垮所有并发请求
        try:
            collected = await asyncio.to_thread(
                self._collect, prompt, requested_model, not agent_mode, tools or None
            )
        except HTTPException as e:
            # anthropic 协议错误体用标准形状（Claude Code 等 SDK 靠它渲染错误）
            if protocol == "anthropic":
                return JSONResponse(
                    status_code=e.status_code,
                    content={
                        "type": "error",
                        "error": {"type": "api_error", "message": str(e.detail)},
                    },
                )
            raise
        if protocol == "anthropic":
            collected = chat_completion_to_anthropic_message(collected, original)
        return JSONResponse(content=collected)

    def _stream(
        self, messages: list[dict[str, Any]], model: str, sanitize: bool = True,
        tools: list[dict[str, Any]] | None = None,
        include_usage: bool = False,
    ) -> AsyncIterator[str]:
        """把 Trae SSE 转成 OpenAI chat.completion.chunk 流。"""
        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        in_thinking = False
        usage: dict[str, Any] | None = None

        def chunk(delta: dict[str, Any], finish: str | None = None) -> str:
            payload = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        def usage_chunk() -> str:
            # OpenAI 流式协议：stream_options.include_usage 时，[DONE] 前须有一个
            # choices 为空、只带 usage 的收尾 chunk。下游客户端（如 ethan）靠它
            # 判定 is_final——缺失会导致 final chunk 永远不到、tool_calls 整体丢失
            payload = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [],
                "usage": usage or {
                    "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                },
            }
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        def error_chunk(msg: str, code: Any = None) -> str:
            # 结构化错误 chunk（与 CodeBuddy 通道 stream_upstream 的错误格式一致）。
            # 不能塞进 content 文本——anthropic 客户端（Claude Code）会把
            # 伪正文当模型输出继续循环；包装层据此转成 event: error
            err: dict[str, Any] = {"message": msg, "type": "upstream_error"}
            if code is not None:
                err["code"] = code
            return f"data: {json.dumps({'error': err}, ensure_ascii=False)}\n\n"

        try:
            raw = send_trae_chat(messages, model, stream=True, base_url=self._base_url)
            cleaner = _StreamLeakCleaner() if sanitize else None
            splitter = (
                _StreamToolCallSplitter(_tool_names(tools)) if tools else None
            )
            dbg_parts: list[str] = []
            for event, data in _parse_sse(raw):
                if event == "error":
                    yield error_chunk(_trae_error_text(data), (data or {}).get("code"))
                    yield "data: [DONE]\n\n"
                    return
                if event == "output":
                    if data.get("reasoning_content"):
                        in_thinking = True
                        yield chunk({"reasoning_content": data["reasoning_content"]})
                    if data.get("response"):
                        text = data["response"]
                        if cleaner is not None:
                            text = cleaner.feed(text)
                        elif splitter is not None:
                            text = splitter.feed(text)
                        if text:
                            dbg_parts.append(text)
                            yield chunk({"role": "assistant", "content": text})
                            in_thinking = False
                elif event == "token_usage":
                    u = data or {}
                    try:
                        usage = {
                            "prompt_tokens": int(u.get("prompt_tokens") or 0),
                            "completion_tokens": int(u.get("completion_tokens") or 0),
                            "total_tokens": int(u.get("total_tokens") or 0),
                        }
                    except (TypeError, ValueError):
                        usage = None
                elif event == "done":
                    break
        except HTTPException as e:
            yield error_chunk(f"trae error {e.status_code}: {e.detail}", e.status_code)
            yield "data: [DONE]\n\n"
            return
        except Exception as e:
            log.error("Trae stream error: %s", e)
            yield error_chunk(f"trae stream error: {e}")
            return

        finish = "stop"
        dbg_calls: list[dict[str, Any]] = []
        if cleaner is not None:
            tail = cleaner.flush()
            if tail:
                dbg_parts.append(tail)
                yield chunk({"role": "assistant", "content": tail})
        elif splitter is not None:
            tail, calls = splitter.flush()
            if tail:
                dbg_parts.append(tail)
                yield chunk({"role": "assistant", "content": tail})
            if calls:
                finish = "tool_calls"
                dbg_calls = calls
                # OpenAI 流式协议：tool_call 必须带 index（客户端靠它合并分片），
                # 首个 delta 带 role；不带 index 会被部分解析器直接丢弃
                yield chunk({"role": "assistant", "tool_calls": [
                    {"index": i, "id": c["id"], "type": "function",
                     "function": c["function"]}
                    for i, c in enumerate(calls)
                ]})
        _debug_dump("debug_trae_response", model=model, stream=True,
                    content="".join(dbg_parts),
                    tool_calls=[
                        {"name": c["function"]["name"],
                         "arguments": c["function"]["arguments"]}
                        for c in dbg_calls
                    ])
        yield chunk({}, finish)
        if include_usage:
            yield usage_chunk()
        yield "data: [DONE]\n\n"

    def _collect(
        self, messages: list[dict[str, Any]], model: str, sanitize: bool = True,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """非流式：聚合 Trae SSE 成完整响应。"""
        reasoning_parts: list[str] = []
        content_parts: list[str] = []

        raw = send_trae_chat(messages, model, stream=False, base_url=self._base_url)
        for event, data in _parse_sse(raw):
            if event == "error":
                raise HTTPException(
                    status_code=502,
                    detail=_trae_error_text(data),
                )
            if event == "output":
                if data.get("reasoning_content"):
                    reasoning_parts.append(data["reasoning_content"])
                if data.get("response"):
                    content_parts.append(data["response"])

        reasoning = "".join(reasoning_parts)
        content = "".join(content_parts)
        # 无 tools 请求也必须有初始值——下方 _debug_dump 无条件引用
        # （实测缺失时非流式无 tools 请求直接 UnboundLocalError -> internal error）
        tool_calls: list[dict[str, Any]] = []
        # 顺序关键：带 tools 的请求必须先解析、后清洗——_sanitize_agent_leak
        # 会把 <tool_call>/</arg_value> 标签全剥掉，先清洗再解析会让解析器
        # 拿到被拆掉结构的残骸（实测 glm-5.3 函数调用表达式因此整段漏进正文）
        if tools:
            content, tool_calls = _parse_tool_calls(content, _tool_names(tools))
        if sanitize:
            content = _sanitize_agent_leak(content)
        _debug_dump("debug_trae_response", model=model, stream=False, content=content,
                    tool_calls=len(tool_calls))
        # 只有 tool_calls 没有正文也是合法响应（agent 直接发起调用），
        # 不注入空响应兜底文案——否则会混进 Anthropic tool_use 消息正文
        if not content and not reasoning and not tool_calls:
            content = "(trae upstream 返回了空响应，未产生任何内容)"
        if reasoning:
            if content:
                content = f"<think>\n{reasoning}\n</think>\n\n{content}"
            else:
                # 只有思维链没有正文时，别把 reasoning 整个丢掉
                content = f"<think>\n{reasoning}\n</think>"

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    **({"tool_calls": tool_calls} if tool_calls else {}),
                },
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": max(1, len(content.encode("utf-8")) // 4),
                "total_tokens": 0,
            },
        }


# ───────────────────────── CLI 入口 ─────────────────────────

def _cli() -> None:
    """命令行工具：查询/领取积分、测试对话。示例：

    python -m buddy_proxy.trae_provider status   # 签到/积分状态
    python -m buddy_proxy.trae_provider claim    # 领取今日签到积分
    python -m buddy_proxy.trae_provider usage    # 权益/用量
    python -m buddy_proxy.trae_provider chat -m glm-5.2 -q "你好"
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="trae_provider",
        description="Trae 账号工具（签到/积分/对话）",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="查询签到/积分状态")
    sub.add_parser("claim", help="领取今日签到积分")
    sub.add_parser("usage", help="查询权益/用量")
    p_chat = sub.add_parser("chat", help="发一条对话测试")
    p_chat.add_argument("-m", "--model", default="glm-5.2", help="模型名")
    p_chat.add_argument("-q", "--query", default="用一句话介绍你自己", help="问题")

    args = parser.parse_args()

    if args.cmd == "status":
        data = fetch_checkin_status()
        print(json.dumps(data, ensure_ascii=False, indent=2))
        if data.get("credits") is not None:
            print(f"\n可用积分: {data['credits']} | 今日已签到: {data.get('checked_in')}")
    elif args.cmd == "claim":
        data = claim_checkin_credits()
        print(json.dumps(data, ensure_ascii=False, indent=2))
        if data.get("code") == 0:
            print(f"\n✅ 签到成功，今日积分 +{data.get('credits_granted', '?')}")
        else:
            print(f"\n签到失败: {data.get('message', data)}")
    elif args.cmd == "usage":
        try:
            token, uid = _auth()
        except Exception:
            work = _load_work_cred() or {}
            token, uid = work.get("access_token", ""), work.get("uid", "")
            if not token:
                print("Trae 未认证：请先运行 trae_work_login.py 登录，或设置 TRAE_TOKEN 环境变量")
                return
        headers = _build_headers(token, uid)
        headers["Accept"] = "application/json"
        headers["X-User-Region"] = "CN"
        req = urllib.request.Request(
            _UG_API_HOST + "/trae/api/v2/pay/ide_user_ent_usage",
            data=b"{}", headers=headers, method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        us = data.get("usage_summary", {})
        print(f"总额度: {us.get('total_amount')} | 已用: {us.get('consumed_amount')} "
              f"({us.get('consumption_ratio', 0) * 100:.1f}%)")
        for p in data.get("user_entitlement_pack_list", []):
            eb = p.get("entitlement_base_info", {})
            print(f"权益包: {p.get('display_desc')} | status={eb.get('ent_status')} "
                  f"| end={eb.get('end_time')} | endpoint={eb.get('available_endpoint')}")
    elif args.cmd == "chat":
        # send_trae_chat 内部会自动路由 Work 通道（有 Work 凭证时），
        # 无需在此重复 if/else 分派
        raw = send_trae_chat(
            [{"role": "user", "content": [{"type": "text", "text": args.query}]}],
            model=args.model, stream=True,
        )
        for event, data in _parse_sse(raw):
            if event == "error":
                # Trae 走 HTTP 200 + event: error（如免费账号撞 4011 限额），
                # 必须显式报错退出，否则打印空白像成功
                print(f"[错误] {_trae_error_text(data)}", file=sys.stderr)
                sys.exit(1)
            if event == "output":
                if data.get("reasoning_content"):
                    print(f"[思考] {data['reasoning_content']}")
                if data.get("response"):
                    print(f"[回答] {data['response']}")


if __name__ == "__main__":
    _cli()
