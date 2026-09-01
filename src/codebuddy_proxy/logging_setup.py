"""日志配置与运行时信息工具函数。"""

from __future__ import annotations

import importlib.metadata
import logging
import logging.handlers
import pathlib
import platform
import time


def setup_logging(log_dir: pathlib.Path) -> logging.Logger:
    """配置滚动日志：按天分片，保留30天。"""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "proxy.log"

    logger = logging.getLogger("codebuddy_proxy")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    handler = logging.handlers.TimedRotatingFileHandler(
        log_file, when="midnight", interval=1, backupCount=30, encoding="utf-8"
    )
    handler.suffix = "%Y-%m-%d"
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def setup_json_logging(log_file: pathlib.Path) -> logging.Logger:
    """配置 JSONL 滚动日志：按天分片，保留30天。"""
    logger = logging.getLogger("codebuddy_proxy.jsonl")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    handler = logging.handlers.TimedRotatingFileHandler(
        log_file, when="midnight", interval=1, backupCount=30, encoding="utf-8"
    )
    handler.suffix = "%Y-%m-%d"
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

    return logger


def now_s() -> int:
    return int(time.time())


def get_runtime_info() -> dict[str, str]:
    try:
        app_version = importlib.metadata.version("codebuddy-proxy")
    except importlib.metadata.PackageNotFoundError:
        app_version = "unknown"
    return {
        "app_version": app_version,
        "system_version": platform.platform(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
    }
