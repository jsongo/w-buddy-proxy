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
    "claude-opus-4-7": "glm-5.3",
    "claude-opus-4-6": "glm-5.3",
    "claude-opus-4-5": "glm-5.3",
    "claude-sonnet-4-6": "glm-5.3",
    "claude-sonnet-4-5": "glm-5.3",
    "claude-sonnet-4": "glm-5.3",
    "claude-3.5-sonnet": "glm-5.3",
    "claude-3.7-sonnet": "glm-5.3",
    "claude-haiku-4-5": "DeepSeek-V4-Flash",
    "mimo-v2.5-pro": "glm-5.3",
    "mimo-v2.5": "glm-5.3",
    "gpt-4o": "DeepSeek-V4-Pro",
    "gpt-4o-mini": "DeepSeek-V4-Flash",
    "gpt-4.1": "DeepSeek-V4-Pro",
    "auto": "glm-5.3",
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
    "T1": ["glm-5.3", "Doubao-Seed-Evolving"],
    "T2": ["glm-5.2", "Doubao-Seed-2.1-Pro", "DeepSeek-V4-Pro",
           "kimi-k2.7-code", "qwen3.8-max"],
    "T3": ["Doubao-Seed-2.1-Turbo", "DeepSeek-V4-Flash", "minimax-m3",
           "kimi-k2.6", "glm-5.1"],
    "T4": ["Doubao-Seed-Code", "glm-5", "glm-5-turbo", "qwen-3.7-plus"],
}

# 部分模型在 solo_work_lite function 下不可用（服务端 4001），需改用 chat_v3。
# 实测：glm-5.1 / Doubao-Seed-Code 仅在 chat_v3 下可路由。
_WORK_FUNCTION_OVERRIDE: dict[str, str] = {
    "glm-5.1": "chat_v3",
    "Doubao-Seed-Code": "chat_v3",
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
    # 孤立 tool_call 标签
    re.compile(r"</?\s*tool_call[^>]*>"),
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
        "<command", "think", "执行过程发生错误",
    )):
        return text
    for pat in _LEAK_RE:
        text = pat.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# 泄漏标签名（流式扣留判断用；大小写不敏感）
_LEAK_TAG_NAMES = ("tool_action", "tool_name", "tool_call", "think", "command")


def _stream_hold_pos(buf: str) -> int:
    """返回应扣留的起始位置（-1 表示整段可安全释放）。

    两种扣留场景：
    1. 存在未闭合的泄漏标签开启（如 <tool_action> 还没等到 </tool_action>）
       ——必须等闭合标签到齐后整块清洗，否则会出现"剥了外壳留下内脏"
       的残骸（下游收到的正是这种，解析器不认）
    2. 尾部的裸 "<" 是某个泄漏标签名的前缀（如 "<tool_ac"）——扣住防止
       标签名被 SSE 分包分裂
    """
    low = buf.lower()
    best = -1
    for tag in _LEAK_TAG_NAMES:
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


def _extract_prompt(messages: list[dict[str, Any]], guard: bool = True) -> list[dict[str, Any]]:
    """OpenAI messages -> Trae messages（content 转成 block 数组）。

    纯对话请求（guard=True）注入 agent 协议压制指令：Trae 的 llm_utils_chat
    端点（solo_work_lite/chat_v3 等 function）服务端会给模型注入"你是带 shell
    工具的 coding agent"的预设，纯对话场景下模型会把 <Command>...</Command>
    之类工具调用语法当正文吐出来（下游 OpenAI 协议客户端不认识）。

    coding agent 请求（guard=False，由 _looks_like_agent_request 判定）不注入：
    下游 system prompt 自己教的工具格式模型会完美遵循，注入反而制造指令冲突。
    """
    out: list[dict[str, Any]] = []
    guard_merged = False
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
        if guard and role == "system" and not guard_merged:
            # 合并进调用方自己的 system 消息，保持单条 system
            out.append({
                "role": "system",
                "content": [{"type": "text", "text": _AGENT_GUARD + "\n\n" + text}],
            })
            guard_merged = True
            continue
        out.append({"role": role, "content": [{"type": "text", "text": text}]})
    if guard and not guard_merged:
        out.insert(0, {
            "role": "system",
            "content": [{"type": "text", "text": _AGENT_GUARD}],
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
            return resp.read().decode("utf-8", errors="replace")
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
    4050: "Trae 请求超时，模型服务资源紧张，请稍后重试",
    4051: "Trae 请求超时，模型服务资源紧张，请稍后重试",
}


def _trae_error_text(data: dict[str, Any]) -> str:
    """把 Trae 错误事件转成友好中文文案。"""
    code = data.get("code")
    msg = data.get("message") or data.get("error") or ""
    if isinstance(code, int) and code in _TRAE_ERROR_HINTS:
        hint = _TRAE_ERROR_HINTS[code]
        return f"{hint}" + (f"（{msg}）" if msg and msg not in hint else "")
    if msg:
        return f"Trae 错误: {msg}"
    return "Trae 未知错误"

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
        # coding agent 请求：不注入 guard、不清洗——下游自己解析工具调用语法
        agent_mode = _looks_like_agent_request(messages, body)

        prompt = _extract_prompt(messages, guard=not agent_mode)
        if not prompt:
            raise HTTPException(status_code=400, detail="no text content")

        if stream:
            return StreamingResponse(
                self._stream(prompt, requested_model, sanitize=not agent_mode),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "close"},
            )
        # 非流式聚合内部是同步 urllib 调用（最长 180s），放线程池执行，
        # 避免阻塞事件循环拖垮所有并发请求
        collected = await asyncio.to_thread(
            self._collect, prompt, requested_model, not agent_mode
        )
        return JSONResponse(content=collected)

    def _stream(
        self, messages: list[dict[str, Any]], model: str, sanitize: bool = True
    ) -> AsyncIterator[str]:
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
            cleaner = _StreamLeakCleaner() if sanitize else None
            for event, data in _parse_sse(raw):
                if event == "error":
                    msg = _trae_error_text(data)
                    yield chunk({"content": f"[{msg}]"})
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
                        if text:
                            yield chunk({"role": "assistant", "content": text})
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
            return

        if cleaner is not None:
            tail = cleaner.flush()
            if tail:
                yield chunk({"role": "assistant", "content": tail})
        yield chunk({}, "stop")
        yield "data: [DONE]\n\n"

    def _collect(
        self, messages: list[dict[str, Any]], model: str, sanitize: bool = True
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
        if sanitize:
            content = _sanitize_agent_leak(content)
        if not content and not reasoning:
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
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
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

    python -m codebuddy_proxy.trae_provider status   # 签到/积分状态
    python -m codebuddy_proxy.trae_provider claim    # 领取今日签到积分
    python -m codebuddy_proxy.trae_provider usage    # 权益/用量
    python -m codebuddy_proxy.trae_provider chat -m glm-5.2 -q "你好"
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
