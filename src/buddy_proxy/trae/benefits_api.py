"""Trae 签到 / 积分 / 权益用量上游 API（api.trae.cn UG 接口）。"""

from __future__ import annotations

import hashlib
import json
import logging
import urllib.error
import urllib.request
from typing import Any

from .credentials import _auth

log = logging.getLogger(__name__)

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


def fetch_ent_usage(token: str = "", account_id: str = "") -> dict[str, Any]:
    """查询权益/额度用量（ide_user_ent_usage：总额度 + 权益包列表）。"""
    if not token:
        token, _ = _auth()
    headers = _build_checkin_headers(token, account_id)
    headers["X-User-Region"] = "CN"
    req = urllib.request.Request(
        _UG_API_HOST + "/trae/api/v2/pay/ide_user_ent_usage",
        data=b"{}", headers=headers, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Trae usage [{e.code}]: {e.read().decode()[:300]}")

