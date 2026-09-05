"""管理页设置持久化：默认启用模型、兜底通道覆盖等。

存放在 ``~/.buddy-proxy/settings.json``（可用 ``BUDDY_PROXY_SETTINGS`` 覆盖），
机器本地配置，不进仓库。启动时由 ``__main__`` 加载进 :class:`ProxyState`，
管理页 POST /ui/api/settings 修改后立即写回并热更新到运行态。
"""

from __future__ import annotations

import json
import os
import pathlib
import time
from typing import Any

_DEFAULT_PATH = "~/.buddy-proxy/settings.json"


def settings_path() -> pathlib.Path:
    return pathlib.Path(
        os.getenv("BUDDY_PROXY_SETTINGS", _DEFAULT_PATH)
    ).expanduser()


def load_settings() -> dict[str, Any]:
    try:
        data = json.loads(settings_path().read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_settings(update: dict[str, Any]) -> dict[str, Any]:
    """合并写入并返回合并后的完整设置。"""
    path = settings_path()
    merged = {**load_settings(), **update, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except Exception:
        pass
    return merged


def normalize_default_model(raw: str) -> str:
    """归一化默认模型字符串：去空白、provider 别名归一。"""
    value = (raw or "").strip()
    if "/" in value:
        prefix, model = value.split("/", 1)
        # workbuddy 是 codebuddy 的旧称
        prefix = {"workbuddy": "codebuddy"}.get(prefix, prefix)
        value = f"{prefix}/{model}"
    return value
