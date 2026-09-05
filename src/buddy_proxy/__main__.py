"""Local OpenAI/Responses/Anthropic compatible proxy for CodeBuddy.

High-performance async implementation with FastAPI + httpx.

Features:
- High concurrency: 1000+ concurrent requests
- Low memory footprint: ~5KB per request
- Robust timeout handling with async iterators

Usage with uv:
    uv run python -m buddy_proxy
    uv run python -m buddy_proxy --desensitize
    uv run python -m buddy_proxy --host 0.0.0.0 --port 8787

模块结构（2026-09 拆分）：
- ``logging_setup``  日志配置与运行时信息工具
- ``state``          全局状态（ProxyState、app）
- ``model_list``     模型列表构建（本地配置 + Codex 格式）
- ``metrics``        请求指标收集（/ui 图表数据源）
- ``settings``       管理页设置持久化（默认启用模型等）
- ``codebuddy_client`` 默认 CodeBuddy 上游客户端
- ``codebuddy_provider`` 默认 CodeBuddy 上游与核心转发逻辑
- ``routes``         所有 /v1/* 代理路由
- ``ui``             /ui 管理页与管理接口
- ``__main__``       本文件：仅保留启动入口 main()
"""

from __future__ import annotations

import argparse
import os
import pathlib

import uvicorn
from fastapi import HTTPException

from buddy_proxy.codebuddy_client import CodeBuddyClient
from buddy_proxy.providers import BaseProvider
from buddy_proxy.doubao_provider import DoubaoProvider
from buddy_proxy.logging_setup import setup_logging, setup_json_logging
from buddy_proxy.metrics import MetricsCollector
from buddy_proxy.benefits import BenefitsManager
from buddy_proxy import settings as settings_mod
from buddy_proxy.state import ProxyState, app
from buddy_proxy.settings import normalize_default_model

# 导入 routes / ui 以注册所有 @app 路由（副作用导入）
from buddy_proxy import routes as _routes  # noqa: F401
from buddy_proxy import ui as _ui  # noqa: F401


def main():
    import buddy_proxy.state as _state

    parser = argparse.ArgumentParser(description="CodeBuddy local API proxy")
    parser.add_argument("--host", default=os.getenv("BUDDY_PROXY_HOST", "127.0.0.1"),
                        help="监听地址")
    parser.add_argument("--port", type=int, default=int(os.getenv("BUDDY_PROXY_PORT", "8787")),
                        help="监听端口")
    parser.add_argument("--endpoint", default=os.getenv("CODEBUDDY_ENDPOINT", "https://copilot.tencent.com"),
                        help="CodeBuddy 后端地址")
    parser.add_argument("--session-file", type=pathlib.Path,
                        help="会话文件路径")
    parser.add_argument("--mock-dir", type=pathlib.Path,
                        help="只使用指定目录中的真实响应 fixture，不访问 CodeBuddy 后端")
    # 默认日志目录：项目根下的 logs/（相对 __file__ 定位，部署后路径稳定）
    _default_log_dir = pathlib.Path(__file__).resolve().parents[2] / "logs"
    default_log_file = pathlib.Path(
        os.getenv(
            "BUDDY_PROXY_LOG_FILE",
            str(_default_log_dir / "buddy-proxy.jsonl"),
        )
    ).expanduser()
    parser.add_argument(
        "--log-file",
        type=pathlib.Path,
        default=default_log_file,
        help="记录完整请求/响应的 JSONL 文件（默认 <项目根>/logs/buddy-proxy.jsonl，可用 BUDDY_PROXY_LOG_FILE 覆盖）",
    )
    parser.add_argument("--desensitize", action="store_true",
                        help="启用脱敏处理，对 system 消息中的敏感词插入零宽空格（缓解审核误拦）")
    parser.add_argument("--optimize-context", action="store_true",
                        help="启用消息压缩优化（仅 /v1/responses），大幅减少 token 使用（适用于 Codex CLI 等长上下文场景）")
    parser.add_argument("--login", action="store_true",
                        help="启动时执行浏览器登录/账户查询")
    parser.add_argument("--no-browser", action="store_true",
                        help="登录时不自动打开浏览器")
    parser.add_argument("--verbose-llm", action="store_true",
                        help="log full LLM request/response content (default: summary only, saves 98%% space)")
    parser.add_argument("--default-model", default=None,
                        help="默认启用模型，形如 zcode/glm-5.3（或裸 glm-5.3 按路由自动匹配通道）；"
                             "客户端请求未带 model 字段时使用。"
                             "首次启动时写入设置文件，此后以 ~/.buddy-proxy/settings.json（管理页可改）为准")
    parser.add_argument("--doubao", action="store_true", default=os.getenv("DOUBAO_ENABLED", "") == "1",
                        help="启用豆包 provider（自研内联，直连豆包工作 App CDP）")
    parser.add_argument("--trae", action="store_true", default=os.getenv("TRAE_ENABLED", "") == "1",
                        help="启用 Trae provider（解密 Trae IDE 登录态，直连底层模型）")
    parser.add_argument("--zcode", action="store_true", default=os.getenv("ZCODE_ENABLED", "") == "1",
                        help="启用 Zcode provider（智谱 GLM，Anthropic 端点直通；"
                             "凭据取 ZCODE_API_KEY / secrets / ~/.zcode/v2/config.json）")
    parser.add_argument("--default-provider", default=os.getenv("PROXY_DEFAULT_PROVIDER", "codebuddy"),
                        help="兜底通道：模型名未命中任何 provider 时转发到哪个通道 "
                             "（codebuddy/zcode/trae/doubao，默认 codebuddy；可用 PROXY_DEFAULT_PROVIDER 覆盖）")
    args = parser.parse_args()
    args.log_file = args.log_file.expanduser()
    # provider 别名归一：workbuddy 是 codebuddy 的旧称，用户配置/命令行都可能敲
    args.default_provider = {"workbuddy": "codebuddy"}.get(args.default_provider, args.default_provider)

    # 设置日志
    log_dir = args.log_file.parent if args.log_file else pathlib.Path("logs")
    logger = setup_logging(log_dir)
    json_logger = setup_json_logging(args.log_file)

    # 初始化客户端
    client = CodeBuddyClient(args.endpoint, session_file=args.session_file)

    # 处理登录
    if args.login:
        client.login(open_browser=not args.no_browser)

    # 初始化额外 provider（ZCode/豆包/Trae 等）
    providers: dict[str, BaseProvider] = {}
    # ⚠️ zcode 必须**最先**注册：trae 的 T1 tier 也提供 glm-5.3 / glm-5.3-flash
    # （小写 id），若 trae 先注册，forward_chat 自动匹配会先命中 trae，glm-*
    # 请求就落不到 zcode coding-plan 通道。providers dict 按插入序遍历匹配。
    if args.zcode:
        from buddy_proxy.zcode_provider import ZcodeProvider

        zcode = ZcodeProvider()
        try:
            zcode.ensure_auth()  # 启动时校验凭证，给出清晰的配置提示
        except HTTPException as exc:
            logger.warning("zcode provider 认证未就绪: %s", exc.detail)
        providers[zcode.id] = zcode
        logger.info("Zcode provider enabled (%s)", zcode.health().get("base_url"))
        print("[Zcode] Enabled (BigModel GLM)")
    else:
        print("[Zcode] Disabled (pass --zcode or ZCODE_ENABLED=1 to enable)")

    if args.doubao:
        doubao = DoubaoProvider()
        providers[doubao.id] = doubao
        logger.info("Doubao provider enabled (CDP)")
        print("[Doubao] Enabled (CDP)")
    else:
        print("[Doubao] Disabled (pass --doubao or DOUBAO_ENABLED=1 to enable)")

    if args.trae:
        from buddy_proxy.trae.provider import TraeProvider

        trae = TraeProvider()
        providers[trae.id] = trae
        logger.info("Trae provider enabled")
        print("[Trae] Enabled")
    else:
        print("[Trae] Disabled (pass --trae or TRAE_ENABLED=1 to enable)")

    # 管理页设置文件里的兜底通道优先于 CLI/环境变量（UI 改动跨重启生效）
    saved_settings = settings_mod.load_settings()
    if saved_settings.get("default_provider"):
        args.default_provider = normalize_default_model(saved_settings["default_provider"])

    # 校验兜底通道：必须是 codebuddy 或已启用的 provider id
    if args.default_provider != "codebuddy" and args.default_provider not in providers:
        logger.warning(
            "default-provider '%s' 未启用，回退为 codebuddy（已启用: %s）",
            args.default_provider, list(providers) or "无",
        )
        args.default_provider = "codebuddy"
    print(f"[Router] Default provider: {args.default_provider}")

    # 创建全局状态
    # 默认模型：CLI --default-model 提供初始值；设置文件里已有的值优先
    default_model = normalize_default_model(args.default_model or "")
    if "default_model" in saved_settings:
        default_model = normalize_default_model(saved_settings.get("default_model") or "")
    if default_model and "default_model" not in saved_settings:
        settings_mod.save_settings({"default_model": default_model})
    if default_model:
        print(f"[Default Model] {default_model} (settings: {settings_mod.settings_path()})")

    metrics = MetricsCollector(args.log_file.parent / "metrics.jsonl")

    _state.proxy_state = ProxyState(
        client=client,
        mock_dir=args.mock_dir,
        log_file=args.log_file,
        enable_desensitize=args.desensitize,
        enable_optimize_context=args.optimize_context,
        verbose_llm=args.verbose_llm,
        logger=logger,
        json_logger=json_logger,
        providers=providers,
        default_provider=args.default_provider,
        default_model=default_model or None,
        metrics=metrics,
    )
    # 打卡管理器要引用 state 本身，构造后挂上
    _state.proxy_state.benefits = BenefitsManager(
        args.log_file.parent / "checkin.jsonl", _state.proxy_state
    )
    _state.proxy_state.write_log(
        "startup",
        host=args.host,
        port=args.port,
    )
    logger.info(
        "Runtime: app_version=%s system_version=%s python_version=%s machine=%s",
        _state.proxy_state.runtime_info["app_version"],
        _state.proxy_state.runtime_info["system_version"],
        _state.proxy_state.runtime_info["python_version"],
        _state.proxy_state.runtime_info["machine"],
    )

    # 启动信息输出到 stdout
    print(f"CodeBuddy proxy listening on http://{args.host}:{args.port}")
    print("Endpoints: /ui /v1/models /v1/chat/completions /v1/responses /v1/messages /health")
    print(f"Admin UI: http://127.0.0.1:{args.port}/ui")

    # 同时记录到日志
    logger.info(f"CodeBuddy proxy listening on http://{args.host}:{args.port}")
    logger.info("Endpoints: /ui /v1/models /v1/chat/completions /v1/responses /v1/messages /health")

    # 启动 uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
