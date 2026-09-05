"""ZCode / BigModel (GLM) provider —— Anthropic 兼容端点直通转发。

上游为智谱 Z.ai / BigModel 的 Anthropic 兼容 coding 端点（与 ZCode CLI 的
bigmodel-coding-plan 通道同款）：

- BigModel（国内）: https://open.bigmodel.cn/api/anthropic  （/v1/messages）
- Z.AI（国际）:     https://api.z.ai/api/anthropic

协议与路由（与 trae/豆包 provider 的多协议约定一致）：
- anthropic（/v1/messages）→ 直通上游 ``{base}/v1/messages``，model 名透传，
  零转换开销；响应（流式/非流式）原样回传。
- openai（/v1/chat/completions）→ GLM Coding Plan 的 OpenAI 端点
  ``https://open.bigmodel.cn/api/coding/paas/v4/chat/completions``（注意 /coding
  前缀：标准 /api/paas/v4 按普通余额计费，Coding Plan 账户 429 1113），请求体透传；
  响应（含 SSE）原样回传。
- responses（/v1/responses）→ 走通用链路（anthropic_adapter 转成 chat 后
  到这里，即 openai 协议路径）。

认证（凭据来源优先级）：
1. 环境变量 ``ZCODE_API_KEY``
2. secrets ``zcode_api_key``（~/.ethan/.secrets/zcode_api_key）
3. 本机 ZCode CLI 配置 ``~/.zcode/v2/config.json`` 中已启用的
   ``builtin:bigmodel-coding-plan`` / ``builtin:zai`` 等 provider 的 apiKey
   （格式 ``<apiKey>.<secretKey>``，即智谱官网 coding-plan API Key）

安全：API key 只在服务端使用，绝不明文进日志；secrets 文件 chmod 600。
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.parse
from pathlib import Path
from typing import Any, AsyncIterator, Iterator, Sequence

import httpx
from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from .providers import BaseProvider

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量与默认模型表
# ---------------------------------------------------------------------------

BIGMODEL_ANTHROPIC_BASE = "https://open.bigmodel.cn/api/anthropic"
ZAI_ANTHROPIC_BASE = "https://api.z.ai/api/anthropic"
# Coding Plan 的 openai 端点带 /coding 前缀（资源包只覆盖它）；标准 /api/paas/v4
# 按普通 API 余额计费，Coding Plan 账户打过去 429 code=1113「余额不足或无可用
# 资源包，请充值」（2026-09 实测）。ZCODE_OPENAI_BASE 可整体覆盖。
BIGMODEL_OPENAI_BASE = "https://open.bigmodel.cn/api/coding/paas/v4"
ZAI_OPENAI_BASE = "https://api.z.ai/api/coding/paas/v4"


def _openai_base_for(anthropic_base: str) -> str:
    """由 anthropic base 推导同域 openai 兼容端点（Coding Plan 的 /coding 变体）。

    key 从 ZCode CLI 配置解析、且启用的是 zai（国际 api.z.ai）通道时，
    key 是 z.ai 域的——openai/responses 请求若还打 open.bigmodel.cn 必然
    401。anthropic 请求走 self._base 没问题，openai 端点必须同域跟随。
    ZCODE_OPENAI_BASE 环境变量可整体覆盖（如官方未来调整端点）。
    """
    override = os.environ.get("ZCODE_OPENAI_BASE", "").strip().rstrip("/")
    if override:
        return override
    host = urllib.parse.urlparse(anthropic_base).netloc.lower()
    return ZAI_OPENAI_BASE if "z.ai" in host else BIGMODEL_OPENAI_BASE

# 与 ZCode CLI 内置模型表一致（~/.zcode/v2/config.json）。
# id 用**小写**（与全站 models_config.json / Claude Code 侧配置一致），
# 这样 forward_chat 的「自动匹配 provider.models()」能命中 zcode，
# 不至于让 glm-* 请求错误落到 codebuddy 兜底通道。
DEFAULT_MODELS: dict[str, str] = {
    "glm-5.3": "GLM-5.3 (thinking, 1M ctx)",
    "glm-5.3-flash": "GLM-5.3-Flash (thinking, multimodal, 1M ctx)",
    "glm-5-turbo": "GLM-5-Turbo (200K ctx)",
}

# 小写 id → 上游正式模型名（ZCode/智谱侧的大小写字面量）。
# anthropic 直通时若上游对 model 名大小写敏感，用它归一化后再转发。
MODEL_NAME_CANONICAL: dict[str, str] = {
    "glm-5.3": "GLM-5.3",
    "glm-5.3-flash": "GLM-5.3-Flash",
    "glm-5-turbo": "glm-5-turbo",
}

_TIMEOUT = httpx.Timeout(connect=15.0, read=600.0, write=60.0, pool=15.0)


def _window_label(reset_ts: float | None, ltype: str | None) -> str:
    """按重置时间推断限额窗口的展示名（5 小时 / 每周 / 更长周期）。"""
    if ltype == "TIME_LIMIT":
        return "MCP 调用（月）"
    if reset_ts:
        delta = reset_ts - time.time()
        if delta < 86400 * 2:
            return "5 小时窗口"
        if delta < 86400 * 10:
            return "每周窗口"
        return f"周期窗口（{time.strftime('%m-%d', time.localtime(reset_ts))} 重置）"
    return ltype or "用量窗口"


def _quota_items(data: dict[str, Any]) -> list[dict[str, Any]]:
    """把 /api/monitor/usage/quota/limit 的 limits 归一化为管理页额度条目。

    按 nextResetTime 升序：最近的窗口排前面（实测 5 小时窗口在前、每周在后）。
    CREDIT_LIMIT：usage=窗口总额度，currentValue=已用；TIME_LIMIT 单位为次数。
    """
    limits = [l for l in (data.get("limits") or []) if isinstance(l, dict)]
    limits.sort(key=lambda l: l.get("nextResetTime") or 0)
    items: list[dict[str, Any]] = []
    for l in limits:
        reset_ms = l.get("nextResetTime")
        reset_ts = reset_ms / 1000 if reset_ms else None
        items.append({
            "label": _window_label(reset_ts, l.get("type")),
            "used": l.get("currentValue"),
            "total": l.get("usage"),
            "remaining": l.get("remaining"),
            "percent": l.get("percentage"),
            "reset_ts": reset_ts,
        })
    return items


# ---------------------------------------------------------------------------
# 凭证解析
# ---------------------------------------------------------------------------

def _secret_file_path() -> Path:
    """secrets 约定路径：~/.ethan/.secrets/zcode_api_key。"""
    return Path.home() / ".ethan" / ".secrets" / "zcode_api_key"


def _load_secret_file() -> str:
    """secrets 约定文件 ~/.ethan/.secrets/zcode_api_key，支持 'name=value' 或裸 value。

    只取首行、且按行切分：secrets 文件若真有多行，把整块文本当 key 发给
    上游必然 401（单行文件不受影响）。按行解析天然规避该问题。
    """
    try:
        for raw in _secret_file_path().read_text().splitlines():
            line = raw.strip()
            if not line:
                continue
            return line.split("=", 1)[1].strip() if "=" in line else line
    except Exception:
        pass
    return ""


def _load_zcode_config_key() -> tuple[str, str]:
    """从本机 ZCode CLI 配置读取已启用 provider 的 apiKey 与 baseURL。

    返回 (api_key, anthropic_base)；enabled 通道优先，其次按配置顺序。
    """
    try:
        cfg = json.loads((Path.home() / ".zcode" / "v2" / "config.json").read_text())
    except Exception:
        return "", ""
    providers = cfg.get("provider") or {}
    ordered = sorted(
        providers.items(),
        key=lambda kv: bool((kv[1] or {}).get("enabled")),
        reverse=True,
    )
    for _pid, p in ordered:
        if not isinstance(p, dict) or p.get("kind") != "anthropic":
            continue
        opts = p.get("options") or {}
        key = opts.get("apiKey") or ""
        base = opts.get("baseURL") or ""
        if key:
            return key, base
    return "", ""


def resolve_credentials() -> tuple[str, str]:
    """解析 (api_key, anthropic_base_url)。来源优先级见模块 docstring。"""
    key = os.environ.get("ZCODE_API_KEY", "").strip()
    if not key:
        key = _load_secret_file()
    base = ""
    if not key:
        key, base = _load_zcode_config_key()
    if not base:
        base = BIGMODEL_ANTHROPIC_BASE
    return key, base.rstrip("/")


def _auth_headers(api_key: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    """上游是 Anthropic 兼容端点，用 x-api-key（与 Anthropic 官方协议一致）。"""
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    if extra:
        headers.update(extra)
    return headers


# ---------------------------------------------------------------------------
# 上游响应处理（直通模式：SSE 原样回传，非流式 JSON 原样回传）
# ---------------------------------------------------------------------------

async def _pass_through_stream(
    client: httpx.AsyncClient,
    response: httpx.Response,
) -> AsyncIterator[bytes]:
    """把上游 SSE/字节流原样泵给客户端；结束后确保连接释放。"""
    try:
        async for chunk in response.aiter_bytes():
            if chunk:
                yield chunk
    finally:
        await response.aclose()


def _upstream_error_response(resp: httpx.Response) -> JSONResponse:
    """把上游错误转成客户端错误响应（透传状态码与错误体，隐藏 key 痕迹）。"""
    try:
        payload = resp.json()
    except Exception:
        payload = {"error": {"message": resp.text[:500], "type": "upstream_error"}}
    return JSONResponse(status_code=resp.status_code, content=payload)


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class ZcodeProvider(BaseProvider):
    id = "zcode"
    name = "ZCode (BigModel GLM, Anthropic 直通)"

    def __init__(self, base_url: str | None = None, api_key: str | None = None):
        if api_key or base_url:
            self._api_key = api_key or ""
            self._base = (base_url or BIGMODEL_ANTHROPIC_BASE).rstrip("/")
        else:
            key, base = resolve_credentials()
            self._api_key = key
            self._base = base
        self._client: httpx.AsyncClient | None = None

    # ---- BaseProvider 接口 ----

    def models(self) -> Sequence[dict[str, Any]]:
        return [
            {
                "id": model_id,
                "object": "model",
                "created": 0,
                "owned_by": self.id,
                "description": desc,
            }
            for model_id, desc in DEFAULT_MODELS.items()
        ]

    def ensure_auth(self) -> None:
        if not self._api_key:
            self._api_key, self._base = resolve_credentials()
        if not self._api_key:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "message": (
                            "zcode 未配置认证：请设置 ZCODE_API_KEY / "
                            "~/.ethan/.secrets/zcode_api_key，或在本机 ZCode CLI "
                            "登录 coding-plan（凭据存于 ~/.zcode/v2/config.json，"
                            "本 provider 会自动读取）"
                        ),
                        "type": "authentication_error",
                    }
                },
            )

    async def forward(
        self,
        body: dict[str, Any],
        protocol: str,
        original: dict[str, Any] | None = None,
    ) -> StreamingResponse | JSONResponse:
        self.ensure_auth()
        stream = bool(body.get("stream", False))

        if protocol == "anthropic":
            # 直通：/v1/messages 的原始请求体（original）就是 anthropic 格式，
            # 原样转发（tools/system/tool_result 等零损耗）；routes 传入的 body
            # 是转换后的 chat 格式，只取它上面已剥过 provider 前缀的 model 名。
            source = original if isinstance(original, dict) and original else body
            upstream_body = {k: v for k, v in source.items() if not k.startswith("_")}
            if isinstance(original, dict) and original and body.get("model"):
                # 路由层可能剥过 "zcode/" 前缀/归一化过大小写，这里统一映射回
                # 上游正式模型名（如 glm-5.3 → GLM-5.3，ZCode CLI 验证过的格式）
                upstream_body["model"] = MODEL_NAME_CANONICAL.get(body["model"], body["model"])
            url = f"{self._base}/v1/messages"
            headers = _auth_headers(self._api_key, {"Accept": "text/event-stream"})
        else:
            # openai / responses：转投 GLM 原生 OpenAI 兼容端点（model 透传）
            upstream_body = {k: v for k, v in body.items() if not k.startswith("_")}
            url = f"{_openai_base_for(self._base)}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream" if stream else "application/json",
            }

        client = await self._get_client()
        req = client.build_request("POST", url, json=upstream_body, headers=headers)
        try:
            resp = await client.send(req, stream=stream)
        except httpx.TimeoutException as exc:
            log.warning("zcode upstream timeout: %s", exc)
            raise HTTPException(status_code=504, detail={
                "error": {"message": "zcode upstream timeout", "type": "timeout"}}
            ) from exc
        except httpx.HTTPError as exc:
            log.warning("zcode upstream error: %s", exc)
            raise HTTPException(status_code=502, detail={
                "error": {"message": "zcode upstream error", "type": "bad_gateway"}}
            ) from exc

        if resp.status_code >= 400:
            if stream:
                # 先把错误体读完再关流：aclose() 会丢弃未读的 body，
                # 之后 _upstream_error_response 的 json()/text() 拿不到
                # 上游真实错误（429 配额 / 401 key 无效都会变成空错误体
                # 甚至 500）。错误体一般很小，先 aread() 成本可忽略。
                await resp.aread()
            try:
                return _upstream_error_response(resp)
            finally:
                await resp.aclose()

        if stream:
            return StreamingResponse(
                _pass_through_stream(client, resp),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "close"},
            )
        try:
            payload = resp.json()
        except Exception as exc:
            raise HTTPException(status_code=502, detail={
                "error": {"message": "zcode upstream returned non-JSON", "type": "bad_gateway"}}
            ) from exc
        return JSONResponse(content=payload)

    # ---- 内部 ----

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=_TIMEOUT)
        return self._client

    def health(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "configured": bool(self._api_key),
            "base_url": self._base,
            "models": list(DEFAULT_MODELS),
        }

    # ---- 额度查询（/ui 管理页消费；同步 httpx，调用方经 asyncio.to_thread 包装） ----

    def quota(self) -> dict[str, Any] | None:
        """查询 GLM Coding Plan 用量窗口（5 小时 / 每周等 CREDIT/TIME 限额）。

        端点由 anthropic base 推导同源 host（open.bigmodel.cn / api.z.ai），
        认证复用 coding-plan API key（Authorization 原样携带，无 Bearer 前缀）。
        """
        key = self._api_key or resolve_credentials()[0]
        if not key:
            raise RuntimeError("zcode 未配置 API key，无法查询额度")
        origin = self._base.split("/api/")[0]
        url = f"{origin}/api/monitor/usage/quota/limit"
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(url, headers={"Authorization": key, "Content-Type": "application/json"})
        if resp.status_code != 200:
            raise RuntimeError(f"quota HTTP {resp.status_code}: {resp.text[:200]}")
        payload = resp.json()
        if not payload.get("success"):
            raise RuntimeError(payload.get("msg") or "quota query failed")
        data = payload.get("data") or {}
        return {"items": _quota_items(data), "level": data.get("level")}


# ---------------------------------------------------------------------------
# 冒烟自测：python -m buddy_proxy.zcode_provider
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    key, base = resolve_credentials()
    if not key:
        print("no api key found (env ZCODE_API_KEY / secrets / ~/.zcode/v2/config.json)")
        raise SystemExit(1)
    print(f"key: {key[:6]}***{key[-4:]}  base: {base}")
    with httpx.Client(timeout=60) as client:
        r = client.post(
            f"{base}/v1/messages",
            headers=_auth_headers(key),
            json={
                "model": "GLM-5.3-Flash",
                "max_tokens": 128,
                "messages": [{"role": "user", "content": "只回复两个字：pong"}],
            },
        )
        print(f"status: {r.status_code}")
        print(r.text[:600])
