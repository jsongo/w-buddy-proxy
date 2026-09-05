"""Trae 常量与配置：URL、版本头、模型映射、环境开关、调试/心跳工具。"""

from __future__ import annotations

import logging
import os
from typing import Any

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

# X-Ide-Version-Code 单独可调（默认 20260906）：2026-09 实测上游按这个头逐模型
# 门控能力——20260401 下新一代模型（glm-5.3 / kimi-k2.7-code / qwen3.8-max 等）
# 在 chat_v3 上整表 4001；glm-5.3 的放行阈值实测落在 20260801~20260815 之间，
# kimi-k2.7-code / Doubao-Seed-2.1-Pro 20260725 即放行。上游会持续按模型发布
# 节奏抬阈值，此值要跟随真实 Trae 客户端版本更新（可用环境变量覆盖）。
_TRAE_IDE_VERSION_CODE = os.environ.get("WB_TRAE_IDE_VERSION_CODE", "20260906")

# 原生 function calling 通道（2026-09 实测全模型可用，glm-5-turbo 除外——不在
# chat_v3 通道，自动走文本协议兜底）。WB_TRAE_NATIVE_TOOLS=0 可整体关闭。
_NATIVE_TOOLS_ENABLED = os.environ.get("WB_TRAE_NATIVE_TOOLS", "1").lower() not in ("0", "false", "off")
_NATIVE_FUNCTION = os.environ.get("WB_TRAE_NATIVE_FUNCTION", "chat_v3")

# 3 级端点回退（与 trae-local-api 一致）
ENDPOINTS = [
    "/api/agent/v3/llm_utils_chat",
    "/api/ide/v1/chat",
    "/api/agent/v3/create_agent_task",
]

# Trae Work 通道同端点最大尝试次数（含首次）：覆盖 IncompleteRead 等瞬态断流
_WORK_CHAT_MAX_ATTEMPTS = 3

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


# 等待上游期间的心跳间隔（秒）；0 = 关闭心跳。
# 背景：Trae 上游是「整段缓冲」模式——send_trae_chat 同步读完整个 SSE 才返回，
# 生成期间客户端收不到任何数据。长生成（深度 review 大报告等）会触发下游
# 单 chunk 超时（实测 Ethan _CHUNK_TIMEOUT=120s → 回合中止、落库空回复）。
# _stream 在等待时按此间隔发一条 reasoning_content 心跳提示，既喂饱下游
# 超时计时器（续命），又让下游 UI 知道中转还在等。45s 意味着 120s 超时窗口
# 内至少有 2 次心跳，单次 SSE 分包延迟也不会误杀；每条 ~30 字节，开销可忽略。
TRAE_HEARTBEAT_INTERVAL = max(0, int(os.environ.get("WB_TRAE_HEARTBEAT_INTERVAL", "45")))


def _heartbeat_text(waited: int) -> str:
    return f"⏳ [trae 中转] 上游模型仍在生成，已等待 {waited}s（流保持存活）…"

