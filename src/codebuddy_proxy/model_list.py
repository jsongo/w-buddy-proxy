"""模型列表构建：静态列表、本地配置加载、远程动态配置、Codex 格式转换。"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Any, Optional

import httpx

from codebuddy_proxy.state import diagnostic, get_state


class RemoteConfigCache:
    """远程配置缓存（带 TTL）- 匹配 VS Code 插件的逻辑"""

    def __init__(self, url: str, ttl: int = 300):
        self.url = url
        self.ttl = ttl
        self._cache: Optional[dict[str, Any]] = None
        self._last_fetch: float = 0

    async def get_config(self) -> dict[str, Any]:
        """获取配置（带缓存）

        匹配插件逻辑:
        - 如果没有 enterpriseId，返回空配置
        - 端点: /console/enterprises/{enterpriseId}/config/models
        - 响应格式: { data: { data: [...models] } }
        - 超时: 5 秒
        - 失败返回空配置（不抛出异常）
        """
        now = time.time()

        # 缓存有效
        if self._cache and (now - self._last_fetch) < self.ttl:
            age = now - self._last_fetch
            diagnostic("remote_config_cache_hit", age_seconds=age, ttl=self.ttl)
            return self._cache

        # 获取远程配置
        state = get_state()

        # 检查是否有 enterpriseId（匹配插件逻辑）
        session = state.client.session
        account = session.get("account", {})
        enterprise_id = account.get("enterpriseId")

        if not enterprise_id:
            # 个人版用户：返回空配置（匹配插件的 if (!enterpriseId) return []）
            diagnostic(
                "remote_config_skipped",
                reason="no enterprise id",
                account_type=account.get("type", "unknown"),
            )
            return {}

        # 企业版用户：调用动态 API
        try:
            # 构造 URL（匹配插件逻辑）
            path = f"/console/enterprises/{enterprise_id}/config/models"
            full_url = self.url.rstrip("/") + path

            diagnostic("remote_config_fetch_attempt", url=full_url, enterprise_id=enterprise_id)

            # 构造请求头（包含认证信息）
            headers = state.client.auth_headers()
            headers["User-Agent"] = "Mozilla/5.0 (compatible; Genie-IDE/1.0)"
            headers["Accept"] = "application/json"

            # 发起请求（超时 5 秒，匹配插件）
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, read=5.0)) as client:
                resp = await client.get(full_url, headers=headers)

                if resp.status_code == 200:
                    # 解析响应（匹配插件：response.data.data）
                    response_data = resp.json()

                    # 提取模型列表
                    models = []
                    if isinstance(response_data, dict):
                        # 尝试 response.data.data 结构
                        data_field = response_data.get("data")
                        if isinstance(data_field, dict):
                            models = data_field.get("data", [])
                        elif isinstance(data_field, list):
                            models = data_field

                    # 标记模型类型（匹配插件逻辑）
                    if models:
                        models = [
                            {
                                **model,
                                "modelType": (
                                    "enterprise"
                                    if (model.get("id") or "").startswith("custom:")
                                    else "built-in"
                                ),
                            }
                            for model in models
                        ]

                    config = {"models": models}
                    self._cache = config
                    self._last_fetch = now

                    diagnostic(
                        "remote_config_fetch_success",
                        url=full_url,
                        models_count=len(models),
                        enterprise_id=enterprise_id,
                    )
                    return config

                else:
                    diagnostic("remote_config_fetch_failed", url=full_url, status_code=resp.status_code)
                    return {}

        except Exception as e:
            # 匹配插件逻辑：捕获异常后返回空数组
            diagnostic("remote_config_fetch_error", url=full_url, error=str(e))
            return {}


def build_model_list_static() -> list[dict[str, Any]]:
    """构建静态模型列表（兜底配置）"""
    # 从 CodeBuddy 扩展 product.json 提取的真实模型列表（2026-07-13 版本）
    # 共 25 个模型，涵盖 DeepSeek、GLM、Kimi、Hunyuan、Claude 等
    return [
        # 默认模型
        {"id": "default", "name": "Default", "vendor": "codebuddy", "max_input": 168000, "max_output": 32000, "tool_call": True, "images": False},

        # GLM 系列
        {"id": "glm-4.7", "name": "GLM-4.7", "vendor": "zhipu", "max_input": 200000, "max_output": 48000, "tool_call": True, "images": False, "reasoning": True, "desc": "GLM-4.7 model, Well-rounded model for everyday use"},
        {"id": "glm-4.6", "name": "GLM-4.6", "vendor": "zhipu", "max_input": 168000, "max_output": 32000, "tool_call": True, "images": False, "desc": "Advanced language model with strong reasoning capabilities"},

        # DeepSeek 系列
        {"id": "deepseek-v3-2-volc", "name": "DeepSeek-V3.2", "vendor": "deepseek", "max_input": 96000, "max_output": 32000, "tool_call": True, "images": False, "reasoning": True, "desc": "DeepSeek-V3.2, good for daily use"},
        {"id": "deepseek-v3-1-volc", "name": "DeepSeek-V3-1-Terminus", "vendor": "deepseek", "max_input": 96000, "max_output": 32000, "tool_call": True, "images": False, "desc": "DeepSeek's flagship model, good for planning, debugging, coding, and more"},
        {"id": "deepseek-v3-1-lkeap", "name": "DeepSeek-V3-1", "vendor": "deepseek", "max_input": 96000, "max_output": 32000, "tool_call": True, "images": False, "desc": "DeepSeek's flagship model. Good for planning, debugging, coding, and more"},
        {"id": "deepseek-v3-1", "name": "DeepSeek-V3.1", "vendor": "deepseek", "max_input": 96000, "max_output": 32000, "tool_call": True, "images": False, "desc": "DeepSeek's flagship model. Good for planning, debugging, coding, and more"},
        {"id": "deepseek-v3-0324-lkeap", "name": "DeepSeek-V3-0324", "vendor": "deepseek", "max_input": 112000, "max_output": 16000, "tool_call": True, "images": False, "desc": "DeepSeek's flagship model, good for planning, debugging, coding, and more"},
        {"id": "deepseek-r1-0528-lkeap", "name": "DeepSeek-R1-0528", "vendor": "deepseek", "max_input": 96000, "max_output": 16000, "tool_call": True, "images": False, "desc": "Open-source reasoning model from DeepSeek, optimised for logic & math"},

        # Kimi 系列
        {"id": "kimi-k2-instruct-taiji", "name": "Kimi-K2", "vendor": "moonshot", "max_input": 31000, "max_output": 8192, "tool_call": True, "images": False},

        # Hunyuan (混元) 系列 - 对话模型
        {"id": "completion-gf", "name": "completion-gf", "vendor": "tencent", "max_input": 200000, "max_output": 8192, "tool_call": True, "images": False},
        {"id": "hunyuan-chat", "name": "Hunyuan-Turbos", "vendor": "tencent", "max_input": 200000, "max_output": 8192, "tool_call": True, "images": False, "desc": "Tencent's lightweight, fast general-purpose model"},
        {"id": "hunyuan-2.0-instruct", "name": "Hunyuan-2.0-Instruct", "vendor": "tencent", "max_input": 128000, "max_output": 16000, "tool_call": True, "images": False, "reasoning": True},

        # Claude 系列
        {"id": "default-1.1", "name": "Claude-3.7-Sonnet", "vendor": "anthropic", "max_input": None, "max_output": 8192, "tool_call": True, "images": True},
        {"id": "default-1.2", "name": "Claude-4.0-Sonnet", "vendor": "anthropic", "max_input": 200000, "max_output": 24000, "tool_call": True, "images": True, "desc": "Great for daily use. Good at most things"},

        # Hunyuan 视觉模型
        {"id": "hunyuan-turbos-vision", "name": "hunyuan-turbos-vision", "vendor": "tencent", "max_input": 16000, "max_output": 16000, "tool_call": True, "images": True},
        {"id": "hunyuan-t1-vision", "name": "hunyuan-turbos-vision", "vendor": "tencent", "max_input": 16000, "max_output": 24000, "tool_call": True, "images": True},

        # 补全模型（仅用于代码补全，不适合对话）
        {"id": "hunyuan-3b", "name": "hunyuan-3b", "vendor": "tencent", "max_input": None, "max_output": 256, "tool_call": False, "images": False},
        {"id": "hunyuan-7b-dense", "name": "hunyuan-7b", "vendor": "tencent", "max_input": None, "max_output": 256, "tool_call": False, "images": False},
        {"id": "codewise-7b-021", "name": "codewise-7b-021", "vendor": "anthropic", "max_input": None, "max_output": 256, "tool_call": False, "images": False},
        {"id": "codewise-completions", "name": "codewise-completions", "vendor": "anthropic", "max_input": None, "max_output": 256, "tool_call": False, "images": False},
        {"id": "deepseek-r1-0528", "name": "deepseek-r1", "vendor": "tencent", "max_input": None, "max_output": 256, "tool_call": False, "images": False},
        {"id": "deepseek-v3-0324-taco-completion", "name": "deepseek-v3-0324", "vendor": "tencent", "max_input": None, "max_output": 256, "tool_call": False, "images": False},
        {"id": "deepseek-v3-0324", "name": "deepseek-v3", "vendor": "tencent", "max_input": None, "max_output": 8192, "tool_call": False, "images": False},
        {"id": "codewise-navi-v1-2-taco", "name": "codewise-navi-v1-2-taco", "vendor": "tencent", "max_input": None, "max_output": 256, "tool_call": False, "images": False},
    ]


def normalize_model_format(remote_model: dict[str, Any]) -> dict[str, Any]:
    """规范化模型格式为 Codex 兼容的内部格式"""
    model_id = remote_model.get("id", "unknown")
    name = remote_model.get("name", model_id)
    vendor = remote_model.get("vendor", "unknown")

    # 提取 token 限制（支持多种字段名；本地配置统一用 max_input/max_output）
    max_input = (
        remote_model.get("max_input")
        or remote_model.get("maxInputTokens")
        or remote_model.get("context_window")
        or remote_model.get("maxAllowedSize")
    )
    max_output = (
        remote_model.get("max_output")
        or remote_model.get("maxOutputTokens")
        or remote_model.get("max_context_window")
    )

    # 提取能力标志（统一规范化为 bool）
    def _flag(*keys: str) -> bool:
        for key in keys:
            value = remote_model.get(key)
            if value is not None and value is not False:
                return bool(value)
        return False

    tool_call = _flag("tool_call", "supportsToolCall", "supportsTools", "supports_parallel_tool_calls")
    images = _flag("images", "supportsImages")
    # reasoning 可能是 bool 或 dict（如 {"effort":"high","summary":"auto"}），统一收敛为 bool
    reasoning = _flag("reasoning", "supportsReasoning")

    # 提取描述（支持中英文）
    description = (
        remote_model.get("descriptionZh")
        or remote_model.get("descriptionEn")
        or remote_model.get("description")
        or remote_model.get("desc")
    )

    return {
        "id": model_id,
        "name": name,
        "vendor": vendor,
        "max_input": max_input,
        "max_output": max_output,
        "tool_call": tool_call,
        "images": images,
        "reasoning": reasoning,
        "desc": description,
        "tags": remote_model.get("tags", []),
        "modelType": remote_model.get("modelType"),
        "credits": remote_model.get("credits"),
    }


def load_models_from_local_config() -> list[dict[str, Any]]:
    """从本地配置文件加载模型列表"""
    config_file = pathlib.Path(__file__).parent / "models_config.json"

    try:
        if not config_file.exists():
            diagnostic("local_config_missing", path=str(config_file))
            return build_model_list_static()

        with open(config_file, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        models = config_data.get("models", [])
        diagnostic("local_config_loaded", models_count=len(models))

        # 规范化每个模型的格式
        normalized_models = []
        for model in models:
            normalized = normalize_model_format(model)
            normalized_models.append(normalized)

        return normalized_models

    except Exception as e:
        diagnostic("local_config_error", error=str(e), path=str(config_file))
        return build_model_list_static()


async def build_model_list_dynamic(remote_config_cache: Optional[RemoteConfigCache]) -> list[dict[str, Any]]:
    """从远程配置构建模型列表（带降级）"""

    try:
        # 获取远程配置
        if not remote_config_cache:
            diagnostic("dynamic_models_disabled", reason="remote_config_cache not initialized")
            return build_model_list_static()

        remote_config = await remote_config_cache.get_config()

        if not remote_config:
            diagnostic("dynamic_models_fallback", reason="empty remote config")
            return build_model_list_static()

        # 提取模型和白名单
        remote_models = remote_config.get("models", [])
        available_model_ids = set(remote_config.get("availableModels", []))

        diagnostic(
            "dynamic_models_fetched",
            remote_models_count=len(remote_models),
            whitelist_count=len(available_model_ids),
        )

        # 创建模型字典（id -> model）
        static_models = build_model_list_static()
        models_dict = {m["id"]: m for m in static_models}

        # 合并远程模型（覆盖静态配置）
        for remote_model in remote_models:
            model_id = remote_model.get("id")
            if model_id:
                models_dict[model_id] = normalize_model_format(remote_model)

        diagnostic("dynamic_models_merged", total_models=len(models_dict))

        # 应用白名单过滤（如果有）
        if available_model_ids:
            filtered_models = []
            for model_id, model in models_dict.items():
                # 保留条件（参考 VS Code 插件逻辑）:
                # 1. 用户自定义模型
                # 2. 企业模型
                # 3. 在白名单中的模型
                tags = model.get("tags", [])
                vendor = model.get("vendor")
                model_type = model.get("modelType")

                if (
                    "custom" in tags
                    or vendor == "user"
                    or model_type == "enterprise"
                    or model_id in available_model_ids
                ):
                    filtered_models.append(model)

            diagnostic(
                "dynamic_models_filtered",
                filtered_count=len(filtered_models),
                total_count=len(models_dict),
            )
            return filtered_models
        else:
            # 无白名单，返回全部
            diagnostic("dynamic_models_no_whitelist", total_count=len(models_dict))
            return list(models_dict.values())

    except Exception as e:
        diagnostic("dynamic_models_error", error=str(e))
        return build_model_list_static()


def model_to_codex_format(m: dict[str, Any]) -> dict[str, Any]:
    """将简化模型对象转换为完整的 Codex ModelInfo 格式"""
    return {
        # 基础标识
        "id": m["id"],
        "slug": m["id"],
        "name": m.get("name", m["id"]),
        "display_name": m.get("name", m["id"]),
        "description": m.get("desc"),
        "description_en": m.get("descriptionEn") or m.get("description_en"),
        "description_zh": m.get("descriptionZh") or m.get("description_zh"),
        "credits": m.get("credits"),
        "tags": m.get("tags", []),
        "object": "model",
        "created": 1720872952,  # 2024-07-13 的时间戳（Unix epoch）
        "owned_by": m.get("vendor", "codebuddy"),
        # 通道归属：codebuddy（默认）/ trae / doubao ...（与 owned_by 厂牌区分）
        "provider": m.get("provider", "codebuddy"),

        # Reasoning 支持
        "default_reasoning_level": None,
        "supported_reasoning_levels": [{"effort": "High", "description": "High"}] if m.get("reasoning") else [],
        "default_reasoning_summary": "auto",
        "supports_reasoning_summary_parameter": True,

        # Shell 和工具能力
        "shell_type": "default",
        "apply_patch_tool_type": None,
        "web_search_tool_type": "text",
        "experimental_supported_tools": [],
        "supports_parallel_tool_calls": m.get("tool_call", False),

        # 可见性和优先级
        "visibility": "list",
        "supported_in_api": True,
        "priority": 1,

        # 上下文窗口
        # context_window = 当前生效的输入上下文（输入 token 上限）
        # max_context_window = 模型支持的最大上下文（也应为输入 token 上限）
        "context_window": m.get("max_input"),
        "max_context_window": m.get("max_input"),
        "auto_compact_token_limit": None,
        "effective_context_window_percent": 95,

        # 输出截断策略
        "truncation_policy": {
            "mode": "bytes",
            "limit": 10000,
        },

        # 多模态支持
        "input_modalities": ["text", "image"] if m.get("images") else ["text"],
        "supports_image_detail_original": False,

        # Verbosity
        "support_verbosity": False,
        "default_verbosity": None,

        # 其他功能开关
        "include_skills_usage_instructions": False,
        "include_plugin_usage_instructions": False,
        "include_apps_usage_instructions": True,

        # 速度层级和服务层级
        "additional_speed_tiers": [],
        "service_tiers": [],
        "default_service_tier": None,

        # 可选元数据
        "availability_nux": None,
        "model_messages": None,

        # 向后兼容的 base_instructions (遗留字段)
        "base_instructions": "You are a helpful AI assistant.",
    }
