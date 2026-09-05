"""Trae IDE 本地存储解密：tc 加密格式（AES-128-CBC + SHA-512）。

从 Trae IDE 的 storage.json 解出 Cloud-IDE-JWT token 等认证字段。
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

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

