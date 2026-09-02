"""Local OpenAI/Responses/Anthropic compatible proxy for CodeBuddy.

High-performance async implementation with FastAPI + httpx.

Features:
- High concurrency: 1000+ concurrent requests
- Low memory footprint: ~5KB per request
- Robust timeout handling with async iterators

Usage with uv:
    uv run python -m codebuddy_proxy
    uv run python -m codebuddy_proxy --desensitize
    uv run python -m codebuddy_proxy --host 0.0.0.0 --port 8787

模块结构（2026-09 拆分）：
- ``logging_setup``  日志配置与运行时信息工具
- ``state``          全局状态（ProxyState、app、proxy_state、remote_config_cache）
- ``model_list``     模型列表构建（静态/本地/远程 + Codex 格式）
- ``codebuddy_provider`` 默认 CodeBuddy 上游与核心转发逻辑
- ``routes``         所有 FastAPI 路由
- ``__main__``       本文件：仅保留启动入口 main()
"""

from __future__ import annotations

import argparse
import os
import pathlib

import uvicorn

from codebuddy_proxy.codebuddy_client_demo import CodeBuddyClient
from codebuddy_proxy.providers import BaseProvider
from codebuddy_proxy.doubao_provider import DoubaoProvider
from codebuddy_proxy.logging_setup import setup_logging, setup_json_logging
from codebuddy_proxy.state import ProxyState, app
from codebuddy_proxy.model_list import RemoteConfigCache

# 导入 routes 以注册所有 @app 路由（副作用导入）
from codebuddy_proxy import routes as _routes  # noqa: F401


def main():
    import codebuddy_proxy.state as _state

    parser = argparse.ArgumentParser(description="CodeBuddy local API proxy")
    parser.add_argument("--host", default=os.getenv("CODEBUDDY_PROXY_HOST", "127.0.0.1"),
                        help="监听地址")
    parser.add_argument("--port", type=int, default=int(os.getenv("CODEBUDDY_PROXY_PORT", "8787")),
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
            "CODEBUDDY_PROXY_LOG_FILE",
            str(_default_log_dir / "codebuddy-proxy.jsonl"),
        )
    ).expanduser()
    parser.add_argument(
        "--log-file",
        type=pathlib.Path,
        default=default_log_file,
        help="记录完整请求/响应的 JSONL 文件（默认 <项目根>/logs/codebuddy-proxy.jsonl，可用 CODEBUDDY_PROXY_LOG_FILE 覆盖）",
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
    parser.add_argument("--static-models", action="store_true",
                        help="使用静态模型列表（默认从远程 API 动态获取）")
    parser.add_argument("--config-cache-ttl", type=int, default=int(os.getenv("CODEBUDDY_CONFIG_CACHE_TTL", "300")),
                        help="远程配置缓存 TTL（秒，默认 300）")
    parser.add_argument("--doubao", action="store_true", default=os.getenv("DOUBAO_ENABLED", "") == "1",
                        help="启用豆包 provider（自研内联，直连豆包工作 App CDP）")
    parser.add_argument("--trae", action="store_true", default=os.getenv("TRAE_ENABLED", "") == "1",
                        help="启用 Trae provider（解密 Trae IDE 登录态，直连底层模型）")
    args = parser.parse_args()
    args.log_file = args.log_file.expanduser()

    # 设置日志
    log_dir = args.log_file.parent if args.log_file else pathlib.Path("logs")
    logger = setup_logging(log_dir)
    json_logger = setup_json_logging(args.log_file)

    # 初始化客户端
    client = CodeBuddyClient(args.endpoint, session_file=args.session_file)

    # 处理登录
    if args.login:
        client.login(open_browser=not args.no_browser)

    # 初始化额外 provider（豆包/Trae 等）
    providers: dict[str, BaseProvider] = {}
    if args.doubao:
        doubao = DoubaoProvider()
        providers[doubao.id] = doubao
        logger.info("Doubao provider enabled (CDP)")
        print("[Doubao] Enabled (CDP)")
    else:
        print("[Doubao] Disabled (pass --doubao or DOUBAO_ENABLED=1 to enable)")

    if args.trae:
        from codebuddy_proxy.trae_provider import TraeProvider

        trae = TraeProvider()
        providers[trae.id] = trae
        logger.info("Trae provider enabled")
        print("[Trae] Enabled")
    else:
        print("[Trae] Disabled (pass --trae or TRAE_ENABLED=1 to enable)")

    # 创建全局状态
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

    # 初始化远程配置缓存（默认启用动态模型列表）
    if not args.static_models:
        # 默认：动态模式
        _state.remote_config_cache = RemoteConfigCache(
            url=args.endpoint,
            ttl=args.config_cache_ttl
        )
        logger.info(f"Dynamic model list enabled: cache_url={args.endpoint}, ttl={args.config_cache_ttl}s")
        print(f"[Dynamic Models] Enabled (endpoint={args.endpoint}, TTL={args.config_cache_ttl}s)")
    else:
        # 显式禁用：静态模式
        logger.info("Using static model list (25 models)")
        print(f"[Static Models] Using 25 hardcoded models")

    # 启动信息输出到 stdout
    print(f"CodeBuddy proxy listening on http://{args.host}:{args.port}")
    print("Endpoints: /v1/models /v1/chat/completions /v1/responses /v1/messages /health")

    # 同时记录到日志
    logger.info(f"CodeBuddy proxy listening on http://{args.host}:{args.port}")
    logger.info("Endpoints: /v1/models /v1/chat/completions /v1/responses /v1/messages /health")

    # 启动 uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
