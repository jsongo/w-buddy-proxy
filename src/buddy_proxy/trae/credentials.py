"""Trae 凭证与请求头：Work (SOLO) 凭证加载、IDE 存储解密回退、两种请求头。

服务端按 X-Ide-Version-Code 门控模型能力（见 config._TRAE_IDE_VERSION_CODE 注释），
两个头构建函数必须使用同一版本码。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from .auth_storage import _extract_auth_fields, find_auth_data
from .config import (
    X_APP_ID,
    _TRAE_APP_VERSION,
    _TRAE_APP_VERSION_CODE,
    _TRAE_IDE_VERSION_CODE,
)

log = logging.getLogger(__name__)

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
        "X-Ide-Version-Code": _TRAE_IDE_VERSION_CODE,
        "X-App-Version-Code": _TRAE_APP_VERSION_CODE,
        "X-Ide-Version-Type": "stable",
        "X-Device-Type": "windows",
        "X-OS-Version": "Windows 11 Pro",
        "X-Device-Brand": "83DG",
        "Request-Traffic-Type": "prod",
        "X-Machine-Id": machine_id,
        "X-Device-Id": device_id,
    }

def _work_headers(work: dict[str, Any]) -> dict[str, str]:
    """Work (SOLO) 通道完整请求头（traework2api headers.go 实测值）。

    关键：必须带 User-Agent: Trae/<ver> + X-Ide-Token 等 SOLO 专属头，
    缺 UA 会被服务端当异常客户端限流（4011）。
    """
    machine_id = work.get("machine_id") or uuid.uuid4().hex
    device_id = work.get("device_id") or hashlib.sha256(machine_id.encode()).hexdigest()[:32]
    return {
        "Content-Type": "application/json",
        "User-Agent": f"Trae/{_TRAE_APP_VERSION}",
        "Authorization": f"Cloud-IDE-JWT {work['access_token']}",
        "X-Cloudide-Token": work["access_token"],
        "X-Ide-Token": work["access_token"],
        "X-Uid": work.get("uid") or "",
        "X-App-Id": X_APP_ID,
        "X-App-Version": "default",
        "X-Ide-Version": _TRAE_APP_VERSION,
        "X-Ide-Version-Code": _TRAE_IDE_VERSION_CODE,
        "X-App-Version-Code": _TRAE_APP_VERSION_CODE,
        "X-Ide-Version-Type": "stable",
        "X-Device-Type": "windows",
        "X-OS-Version": "Windows 11 Pro",
        "X-Device-Brand": "83DG",
        "Request-Traffic-Type": "prod",
        "X-Machine-Id": machine_id,
        "X-Device-Id": device_id,
    }

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

