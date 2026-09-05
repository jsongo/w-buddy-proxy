"""管理 UI：/ui 页面与 /ui/api/* 管理接口。

功能：
- 按 provider 分组的模型列表，一键「设为默认启用模型」（settings.py 持久化）
- 每个模型一键测试：发一条 "hi"，返回延迟 / token / 回复预览
- 按 provider/模型维度聚合的请求统计（metrics.py），含近 14 天图表与最近请求
- provider 健康状态总览

安全约定：/ui/api/* 仅允许本机（127.0.0.1 / ::1）访问；如确需从局域网打开
管理页操作，设置环境变量 ``BUDDY_PROXY_ADMIN_OPEN=1`` 放开（自担风险）。
/v1/* 代理端点不受此限制。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
from typing import Any

from fastapi import HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from buddy_proxy.state import app, get_state, _get_state_or_none
from buddy_proxy.model_list import load_models_from_local_config
from buddy_proxy.codebuddy_provider import forward_chat
from buddy_proxy.benefits import BenefitsManager, read_checkin_settings
from buddy_proxy import settings as settings_mod

# 一键测试发送的内容与 token 上限（够穿透 thinking 模型的少量预算）
TEST_PROMPT = "hi"
TEST_MAX_TOKENS = 256
TEST_TIMEOUT_S = 120

_LOCAL_HOSTS = {"127.0.0.1", "::1", "testclient"}


def _ensure_local(request: Request) -> None:
    """管理接口仅限本机访问（防 LAN 内误触计费请求 / 篡改配置）。"""
    if os.getenv("BUDDY_PROXY_ADMIN_OPEN") == "1":
        return
    host = request.client.host if request.client else ""
    if host not in _LOCAL_HOSTS:
        raise HTTPException(
            status_code=403,
            detail={"error": {"message": "管理接口仅限本机访问；如需放开请设 BUDDY_PROXY_ADMIN_OPEN=1"}},
        )


def _err_text(detail: Any) -> str:
    if isinstance(detail, dict):
        detail = (detail.get("error") or {}).get("message") or detail
    return str(detail)


# ---------------------------------------------------------------------------
# 模型分组（provider -> models）
# ---------------------------------------------------------------------------

def _codebuddy_models() -> list[dict[str, Any]]:
    models = []
    for m in load_models_from_local_config():
        models.append({
            "id": m.get("id"),
            "name": m.get("name") or m.get("id"),
            "vendor": m.get("vendor"),
            "credits": m.get("credits"),
            "tags": m.get("tags", []),
            "context_window": m.get("max_input"),
            "reasoning": bool(m.get("reasoning")),
        })
    return models


def _provider_models(provider: Any) -> list[dict[str, Any]]:
    models = []
    for m in provider.models():
        models.append({
            "id": m.get("id"),
            "name": m.get("description") or m.get("name") or m.get("id"),
            "vendor": provider.id,
            "credits": m.get("credits"),
            "tier": m.get("tier"),
            "tags": m.get("tags", []),
            "context_window": m.get("context_window") or m.get("max_input"),
            "reasoning": bool(m.get("reasoning")),
        })
    return models


def _model_groups(state: Any) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []

    # 默认 CodeBuddy 通道（模型来自 models_config.json）
    auth = {} if state.mock_dir is not None else (state.client.session.get("auth") or {})
    expires = int(auth.get("expiresAt") or 0)
    groups.append({
        "id": "codebuddy",
        "name": "CodeBuddy（默认通道）",
        "enabled": True,
        "health": {
            "authenticated": bool(auth.get("accessToken")),
            "token_valid": not expires or expires > int(time.time() * 1000),
        },
        "models": _codebuddy_models(),
    })

    for pid, p in getattr(state, "providers", {}).items():
        try:
            health = p.health()
        except Exception:
            health = {}
        groups.append({
            "id": pid,
            "name": p.name,
            "enabled": True,
            "health": health,
            "models": _provider_models(p),
        })
    return groups


def _validate_model(provider_id: str, model_id: str, state: Any) -> None:
    """校验 (provider, model) 组合真实可用，否则 400。"""
    for group in _model_groups(state):
        if group["id"] == provider_id:
            if any(m["id"] == model_id for m in group["models"]):
                return
            raise HTTPException(
                status_code=400,
                detail={"error": {"message": f"模型 {model_id} 不在 {provider_id} 通道的模型列表中"}},
            )
    raise HTTPException(
        status_code=400,
        detail={"error": {"message": f"未知 provider: {provider_id}（未启用或不存在）"}},
    )


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.get("/ui/api/overview")
async def ui_overview(request: Request):
    _ensure_local(request)
    state = get_state()
    auth = {} if state.mock_dir is not None else (state.client.session.get("auth") or {})
    providers_health = {pid: p.health() for pid, p in getattr(state, "providers", {}).items()}

    default_model = getattr(state, "default_model", None)
    provider = model = None
    if default_model and "/" in default_model:
        provider, model = default_model.split("/", 1)
    elif default_model:
        model = default_model

    return {
        "uptime_seconds": int(time.time() - state.started_at),
        "authenticated": bool(auth.get("accessToken")),
        "providers": providers_health,
        "default_provider": getattr(state, "default_provider", "codebuddy"),
        "default_model": {"provider": provider, "model": model, "raw": default_model},
        "runtime": getattr(state, "runtime_info", {}),
    }


@app.get("/ui/api/models")
async def ui_models(request: Request):
    _ensure_local(request)
    state = get_state()
    groups = _model_groups(state)

    # 附加每个 (provider, model) 的请求统计
    metrics = getattr(state, "metrics", None)
    stat_map: dict[tuple[str, str], dict[str, Any]] = {}
    if metrics is not None:
        for m in metrics.snapshot(days=14)["models"]:
            stat_map[(m["provider"], m["model"])] = m

    default_model = getattr(state, "default_model", None) or ""
    for group in groups:
        for m in group["models"]:
            st = stat_map.get((group["id"], m["id"])) or {}
            m["stats"] = {
                "count": st.get("count", 0),
                "errors": st.get("errors", 0),
                "avg_ms": st.get("avg_ms", 0),
                "last_ts": st.get("last_ts", 0),
            }
            m["is_default"] = default_model in (f"{group['id']}/{m['id']}", m["id"])
    return {"groups": groups, "default_model": default_model}


@app.get("/ui/api/stats")
async def ui_stats(request: Request):
    _ensure_local(request)
    state = get_state()
    metrics = getattr(state, "metrics", None)
    if metrics is None:
        return {"models": [], "daily": [], "recent": [], "summary": {}}
    return metrics.snapshot(days=14)


@app.get("/ui/api/benefits")
async def ui_benefits(request: Request):
    """打卡状态 + 打卡日历 + 各通道额度（带 5 分钟缓存，避免频打上游）。"""
    _ensure_local(request)
    state = get_state()
    manager = getattr(state, "benefits", None)
    if manager is None:
        return {"providers": [], "calendar": [], "auto_checkin": False,
                "checkin_time": "09:30", "checkin_enabled_providers": []}
    return await manager.snapshot()


@app.post("/ui/api/checkin")
async def ui_checkin(request: Request):
    """立即打卡：向上游领取今日签到积分并记录历史。"""
    _ensure_local(request)
    state = get_state()
    manager = getattr(state, "benefits", None)
    if manager is None:
        raise HTTPException(status_code=503, detail={"error": {"message": "打卡功能未初始化"}})
    body = await request.json()
    provider_id = (body.get("provider") or "").strip()
    if not provider_id:
        raise HTTPException(status_code=400, detail={"error": {"message": "缺少 provider"}})
    return await manager.claim_now(provider_id)


# 自动打卡后台循环随应用启停（uvicorn 生命周期）。
# 启动失败只记日志，不阻断代理本身。
async def _start_benefits_loop() -> None:
    try:
        state = _get_state_or_none()
        manager = getattr(state, "benefits", None) if state else None
        if isinstance(manager, BenefitsManager):
            manager.start()
    except Exception as exc:
        print(f"[Benefits] auto checkin loop failed to start: {exc}")


async def _stop_benefits_loop() -> None:
    try:
        state = _get_state_or_none()
        manager = getattr(state, "benefits", None) if state else None
        if isinstance(manager, BenefitsManager):
            await manager.stop()
    except Exception:
        pass


app.router.on_startup.append(_start_benefits_loop)
app.router.on_shutdown.append(_stop_benefits_loop)


@app.get("/ui/api/settings")
async def ui_settings_get(request: Request):
    _ensure_local(request)
    state = get_state()
    cfg = read_checkin_settings()
    return {
        "default_model": getattr(state, "default_model", None),
        "default_provider": getattr(state, "default_provider", "codebuddy"),
        "auto_checkin": cfg["auto_checkin"],
        "checkin_time": cfg["checkin_time"],
        "path": str(settings_mod.settings_path()),
    }


@app.post("/ui/api/settings")
async def ui_settings_post(request: Request):
    _ensure_local(request)
    state = get_state()
    body = await request.json()

    update: dict[str, Any] = {}
    if "default_model" in body:
        raw = settings_mod.normalize_default_model(body.get("default_model") or "")
        if raw:
            if "/" in raw:
                provider_id, model_id = raw.split("/", 1)
                _validate_model(provider_id, model_id, state)
                # 指定了通道的默认模型顺带把兜底通道对齐（显式前缀路由优先级一致）
                update["default_provider"] = provider_id
            elif not _model_exists_anywhere(raw, state):
                raise HTTPException(
                    status_code=400,
                    detail={"error": {"message": f"裸模型 id {raw} 未命中任何已启用通道的模型列表"}},
                )
            update["default_model"] = raw
        else:
            # 清空默认模型：恢复「按客户端请求原样路由」
            update["default_model"] = ""
    if body.get("default_provider"):
        pid = body["default_provider"]
        if pid != "codebuddy" and pid not in getattr(state, "providers", {}):
            raise HTTPException(
                status_code=400,
                detail={"error": {"message": f"provider {pid} 未启用"}},
            )
        update["default_provider"] = pid
    if "auto_checkin" in body:
        update["auto_checkin"] = bool(body["auto_checkin"])
    if "checkin_time" in body:
        raw_time = str(body["checkin_time"] or "").strip()
        if not re.fullmatch(r"([01]?\d|2[0-3]):[0-5]\d", raw_time):
            raise HTTPException(
                status_code=400,
                detail={"error": {"message": f"打卡时间格式应为 HH:MM，收到 {raw_time!r}"}},
            )
        update["checkin_time"] = raw_time

    if not update:
        raise HTTPException(status_code=400, detail={"error": {"message": "没有可更新的设置字段"}})

    saved = settings_mod.save_settings(update)
    # 热更新运行态（"" 与 None 都视为未设置）
    if "default_model" in update:
        state.default_model = update["default_model"] or None
    if "default_provider" in update:
        state.default_provider = update["default_provider"]
    return {"ok": True, "settings": {k: saved.get(k) for k in ("default_model", "default_provider")}}


@app.post("/ui/api/test")
async def ui_test(request: Request):
    """一键测试：向指定 (provider, model) 发一条 "hi"，返回延迟与回复预览。"""
    _ensure_local(request)
    get_state()  # 未初始化时抛 503
    body = await request.json()
    provider = (body.get("provider") or "").strip()
    model = (body.get("model") or "").strip()
    if not model:
        raise HTTPException(status_code=400, detail={"error": {"message": "缺少 model"}})
    prompt = (body.get("prompt") or TEST_PROMPT).strip() or TEST_PROMPT

    full_model = f"{provider}/{model}" if provider else model
    chat_body = {
        "model": full_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": TEST_MAX_TOKENS,
    }

    started = time.time()
    try:
        resp = await asyncio.wait_for(forward_chat(chat_body, "openai"), timeout=TEST_TIMEOUT_S)
    except asyncio.TimeoutError:
        return {"ok": False, "latency_ms": _ms(started),
                "error": f"测试超时（>{TEST_TIMEOUT_S}s），通道可能未就绪或上游无响应"}
    except HTTPException as exc:
        return {"ok": False, "status": exc.status_code, "latency_ms": _ms(started),
                "error": _err_text(exc.detail)}
    except Exception as exc:
        return {"ok": False, "latency_ms": _ms(started), "error": str(exc)[:300]}

    latency_ms = _ms(started)
    try:
        payload = json.loads(resp.body)
    except Exception:
        return {"ok": False, "status": resp.status_code, "latency_ms": latency_ms,
                "error": "上游返回了无法解析的响应"}

    if resp.status_code >= 400 or payload.get("error"):
        err = payload.get("error")
        message = err.get("message") if isinstance(err, dict) else str(err)
        return {"ok": False, "status": resp.status_code, "latency_ms": latency_ms,
                "error": message or f"HTTP {resp.status_code}"}

    choices = payload.get("choices") or []
    message = (choices[0].get("message") or {}) if choices else {}
    content = message.get("content")
    if isinstance(content, list):  # 兼容分块 content
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return {
        "ok": True,
        "latency_ms": latency_ms,
        "model": payload.get("model") or model,
        "content": (content or "").strip()[:600] or "(空回复)",
        "finish_reason": (choices[0].get("finish_reason") if choices else None),
        "usage": payload.get("usage") or {},
    }


def _ms(started: float) -> int:
    return round((time.time() - started) * 1000)


@app.get("/ui", response_class=HTMLResponse)
async def ui_page():
    # no-cache：页面随代码更新，别让浏览器拿旧缓存（管理页无性能顾虑）
    return HTMLResponse(content=_PAGE_HTML, headers={"Cache-Control": "no-cache"})


@app.get("/")
async def ui_root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/ui")


# ---------------------------------------------------------------------------
# 页面（单文件、零依赖，无外链 CDN）
# ---------------------------------------------------------------------------

_PAGE_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Buddy Proxy 控制台</title>
<style>
  :root {
    --bg: #0e1116; --card: #161b23; --card2: #1c232e; --line: #262f3c;
    --text: #e6e9ee; --muted: #8b96a5; --accent: #4f8cff; --ok: #3ecf8e;
    --warn: #f0b429; --err: #ff6b6b;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text);
         font: 14px/1.55 -apple-system, "PingFang SC", "Segoe UI", "Microsoft YaHei", sans-serif; }
  a { color: var(--accent); text-decoration: none; }
  .wrap { max-width: 1180px; margin: 0 auto; padding: 18px 24px 60px; }
  .topbar { position: sticky; top: 0; z-index: 30; background: rgba(14,17,22,.94);
            backdrop-filter: blur(8px); border-bottom: 1px solid var(--line); }
  .topbar-inner { max-width: 1180px; margin: 0 auto; padding: 12px 24px 0; }
  header { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; margin: 0 0 2px; }
  .tabs { display: flex; gap: 2px; margin-top: 8px; overflow-x: auto; }
  .tab { border: none; background: transparent; color: var(--muted); padding: 8px 14px 9px;
         font-size: 13.5px; border-radius: 8px 8px 0 0; border-bottom: 2px solid transparent;
         white-space: nowrap; }
  .tab:hover { color: var(--text); border-color: transparent; }
  .tab.active { color: var(--accent); border-bottom-color: var(--accent);
                background: rgba(79,140,255,.07); }
  .tab .cnt { font-size: 11px; background: var(--card2); border: 1px solid var(--line);
              border-radius: 999px; padding: 0 7px; margin-left: 5px; color: var(--muted); }
  .tab .cnt.ok { color: var(--ok); border-color: rgba(62,207,142,.4); }
  .tab .cnt.warn { color: var(--warn); border-color: rgba(240,180,41,.4); }
  .page { display: none; }
  .page.active { display: block; animation: fade .18s ease; }
  .page h2:first-child { margin-top: 0; }
  @keyframes fade { from { opacity: 0; } to { opacity: 1; } }
  header h1 { font-size: 19px; margin: 0; letter-spacing: .3px; }
  header .sub { color: var(--muted); font-size: 12.5px; }
  .pill { display: inline-flex; align-items: center; gap: 6px; padding: 3px 11px;
          border-radius: 999px; background: var(--card2); border: 1px solid var(--line);
          font-size: 12.5px; color: var(--muted); }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--ok); }
  .dot.off { background: var(--err); }
  .spacer { flex: 1; }
  button { cursor: pointer; border: 1px solid var(--line); background: var(--card2);
           color: var(--text); border-radius: 8px; padding: 5px 12px; font-size: 12.5px; }
  button:hover { border-color: var(--accent); color: #fff; }
  button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
  button.primary:hover { filter: brightness(1.1); }
  button.ghost { background: transparent; }
  button:disabled { opacity: .45; cursor: not-allowed; }
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin-bottom: 14px; }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px; }
  .card .k { color: var(--muted); font-size: 12px; margin-bottom: 4px; }
  .card .v { font-size: 19px; font-weight: 600; }
  .card .v small { font-size: 12px; color: var(--muted); font-weight: 400; }
  .card .v .chip { display: inline-block; font-size: 11px; padding: 1px 8px; border-radius: 6px;
                   background: rgba(79,140,255,.15); color: var(--accent); margin-right: 6px;
                   vertical-align: 2px; }
  h2 { font-size: 14.5px; margin: 26px 0 10px; color: var(--text); }
  h2 .sub { color: var(--muted); font-size: 12px; font-weight: 400; margin-left: 8px; }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  @media (max-width: 900px) { .grid2 { grid-template-columns: 1fr; } }
  .chart-card { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px; }
  .legend { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 8px; font-size: 12px; color: var(--muted); }
  .legend i { display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 5px; }
  table { width: 100%; border-collapse: separate; border-spacing: 0; }
  th { text-align: left; font-weight: 600; font-size: 12px; letter-spacing: .02em;
       padding: 9px 10px; white-space: nowrap; }
  td { padding: 8px 10px; border-bottom: 1px solid rgba(38,47,60,.55); vertical-align: middle; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: rgba(79,140,255,.04); }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12.5px; }
  .muted { color: var(--muted); }
  /* 模型分组：注意不能用 overflow:hidden（会禁掉内部 position:sticky 吸顶） */
  .group { background: var(--card); border: 1px solid var(--line); border-radius: 12px;
           margin-bottom: 14px; }
  .group-head { position: sticky; top: var(--top-h, 0px); z-index: 12;
                display: flex; align-items: center; gap: 10px; padding: 11px 16px;
                background: var(--card2); border-bottom: 1px solid var(--line);
                border-radius: 12px 12px 0 0; cursor: pointer; user-select: none; }
  .group-head:hover { background: #1d2531; }
  .group-head .name { font-weight: 600; }
  .chev { display: inline-block; transition: transform .15s ease; color: var(--muted);
          font-size: 11px; margin-left: 2px; }
  .group.collapsed .chev { transform: rotate(-90deg); }
  .group.collapsed table { display: none; }
  /* 分组表头（列头）：吸在应用顶栏下方，本表滚完即让位给下一组 */
  .group thead th { position: sticky; top: calc(var(--top-h, 0px) + var(--grp-h, 47px));
                    z-index: 11; cursor: pointer; background: #12161d; color: #d3dae3;
                    border-bottom: 1px solid #2d3846; }
  .group thead th:hover { color: var(--accent); }
  .sec-head { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-top: 22px; }
  .sec-head:first-child { margin-top: 0; }
  .sec-head h2 { margin: 0; }
  .sec-head .sub { color: var(--muted); font-size: 12px; font-weight: 400; margin-left: 8px; }
  .tag { font-size: 11px; padding: 1px 8px; border-radius: 6px; background: var(--bg);
         color: var(--muted); border: 1px solid var(--line); }
  .tag.ok { color: var(--ok); border-color: rgba(62,207,142,.35); }
  .tag.bad { color: var(--err); border-color: rgba(255,107,107,.35); }
  .badge-default { font-size: 11px; color: var(--ok); border: 1px solid rgba(62,207,142,.4);
                   padding: 1px 8px; border-radius: 6px; white-space: nowrap; }
  .bar-wrap { background: var(--bg); border-radius: 4px; height: 8px; overflow: hidden; min-width: 70px; }
  .bar { height: 100%; background: var(--accent); border-radius: 4px; }
  .num { text-align: right; white-space: nowrap; }
  .empty { color: var(--muted); text-align: center; padding: 26px 0; font-size: 13px; }
  /* 打卡 & 额度 */
  .cal { display: grid; grid-template-columns: repeat(7, 1fr); gap: 5px; margin: 12px 0 4px; }
  .cal .dow { display: flex; align-items: center; justify-content: center;
              font-size: 10.5px; color: var(--muted); }
  .cal .day { aspect-ratio: 1; border-radius: 6px; background: var(--bg); border: 1px solid var(--line);
              display: flex; align-items: center; justify-content: center; font-size: 10.5px; color: var(--muted); }
  .cal .day.blank { visibility: hidden; }
  .cal .day.hit { background: rgba(62,207,142,.16); border-color: rgba(62,207,142,.55); color: var(--ok); }
  .cal .day.today { box-shadow: 0 0 0 1.5px var(--accent); }
  .cal-head { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .cal-head label { display: inline-flex; align-items: center; gap: 6px; font-size: 13px; color: var(--text); }
  .cal-head input[type="time"] { background: var(--card2); border: 1px solid var(--line); color: var(--text);
                                 border-radius: 6px; padding: 3px 8px; font-size: 12.5px; }
  .benefit-row { display: flex; align-items: center; gap: 10px; padding: 9px 0;
                 border-bottom: 1px solid rgba(38,47,60,.55); }
  .benefit-row:last-child { border-bottom: none; }
  .grow { flex: 1; }
  .qitem { margin: 9px 0; }
  .qitem .qhead { display: flex; justify-content: space-between; gap: 8px; font-size: 12px;
                  color: var(--muted); margin-bottom: 4px; }
  .qbar { background: var(--bg); height: 8px; border-radius: 4px; overflow: hidden; }
  .qbar > div { height: 100%; border-radius: 4px; }
  /* 弹窗 */
  .overlay { position: fixed; inset: 0; background: rgba(5,8,12,.62); display: none;
             align-items: center; justify-content: center; z-index: 50; }
  .overlay.show { display: flex; }
  .modal { background: var(--card); border: 1px solid var(--line); border-radius: 14px;
           width: min(620px, calc(100vw - 40px)); max-height: 80vh; overflow: auto; padding: 20px 22px; }
  .modal h3 { margin: 0 0 12px; font-size: 15px; }
  .kv { display: grid; grid-template-columns: 96px 1fr; gap: 5px 12px; font-size: 13px; margin-bottom: 10px; }
  .kv .k { color: var(--muted); }
  pre.view { background: var(--bg); border: 1px solid var(--line); border-radius: 8px;
             padding: 12px; white-space: pre-wrap; word-break: break-word; max-height: 300px;
             overflow: auto; font-size: 12.5px; margin: 0; }
  .spin { width: 18px; height: 18px; border: 2px solid var(--line); border-top-color: var(--accent);
          border-radius: 50%; animation: r 0.8s linear infinite; display: inline-block;
          vertical-align: -4px; margin-right: 8px; }
  @keyframes r { to { transform: rotate(360deg); } }
  #toast { position: fixed; bottom: 26px; left: 50%; transform: translateX(-50%);
           background: var(--card2); border: 1px solid var(--line); color: var(--text);
           padding: 9px 18px; border-radius: 10px; font-size: 13px; display: none; z-index: 99; }
  #toast.err { border-color: rgba(255,107,107,.5); color: #ffb3b3; }
  #cal-tip { position: fixed; z-index: 60; display: none; pointer-events: none;
             background: #0b0e13; border: 1px solid #2d3846; color: var(--text);
             padding: 8px 11px; border-radius: 8px; font-size: 12px;
             box-shadow: 0 6px 18px rgba(0,0,0,.45); min-width: 130px; }
  #cal-tip .t-date { color: var(--muted); margin-bottom: 4px; }
  #cal-tip .t-p { display: flex; align-items: center; gap: 6px; margin: 2px 0; }
  #cal-tip .t-p i { width: 8px; height: 8px; border-radius: 2px; display: inline-block; flex: none; }
</style>
</head>
<body>
<div class="topbar">
  <div class="topbar-inner">
    <header>
      <h1>🤝 Buddy Proxy 控制台</h1>
      <span class="pill"><span class="dot" id="status-dot"></span><span id="status-text">加载中…</span></span>
      <span class="sub" id="listen-hint"></span>
      <span class="spacer"></span>
      <button class="ghost" onclick="loadAll(true)">↻ 刷新</button>
    </header>
    <nav class="tabs">
      <button class="tab active" data-tab="overview">📊 总览</button>
      <button class="tab" data-tab="models">🧩 模型<span class="cnt" id="cnt-models"></span></button>
      <button class="tab" data-tab="benefits">✅ 打卡 &amp; 额度<span class="cnt" id="cnt-benefits"></span></button>
    </nav>
  </div>
</div>

<div class="wrap">
  <section class="page active" id="page-overview">
    <div class="cards" id="cards"></div>

    <h2>请求趋势 <span class="sub">近 14 天，按通道堆叠</span></h2>
    <div class="grid2">
      <div class="chart-card"><div id="chart-daily"></div><div class="legend" id="legend-daily"></div></div>
      <div class="chart-card"><div id="chart-models"></div></div>
    </div>

    <h2>最近请求 <span class="sub">最多 50 条，进程内滚动</span></h2>
    <div class="chart-card" style="padding: 4px 8px;">
      <table>
        <thead><tr><th>时间</th><th>通道</th><th>模型</th><th>协议</th><th class="num">耗时</th><th class="num">Tokens</th><th>状态</th></tr></thead>
        <tbody id="recent"><tr><td colspan="7" class="empty">加载中…</td></tr></tbody>
      </table>
    </div>
  </section>

  <section class="page" id="page-models">
    <div class="sec-head">
      <h2>模型 <span class="sub">点击通道条/表头可折叠分组 · 「测试」发送一条 hi</span></h2>
      <span class="spacer"></span>
      <span class="muted" style="font-size:12px" id="model-summary"></span>
      <button class="ghost" onclick="setAllGroups(false)">全部展开</button>
      <button class="ghost" onclick="setAllGroups(true)">全部折叠</button>
    </div>
    <div id="groups"><div class="empty">加载中…</div></div>
  </section>

  <section class="page" id="page-benefits">
    <h2>打卡 &amp; 额度 <span class="sub">自动打卡 · 最近 35 天 · 按通道额度</span></h2>
    <div class="grid2">
      <div class="chart-card">
        <div class="cal-head">
          <label><input type="checkbox" id="auto-checkin"> 自动打卡</label>
          <input type="time" id="checkin-time" value="09:30">
          <button onclick="saveCheckinSettings()">保存</button>
          <span class="muted" style="font-size:12px" id="checkin-hint"></span>
        </div>
        <div class="cal" id="cal"><div class="empty" style="grid-column:1/-1">加载中…</div></div>
        <div id="checkin-rows"></div>
      </div>
      <div id="quota-list"><div class="chart-card"><div class="empty">加载中…</div></div></div>
    </div>
  </section>
</div>

<div class="overlay" id="overlay">
  <div class="modal">
    <h3 id="modal-title">测试</h3>
    <div id="modal-body"><span class="spin"></span>调用上游中…</div>
    <div style="margin-top: 14px; text-align: right;">
      <button onclick="closeModal()">关闭</button>
    </div>
  </div>
</div>
<div id="toast"></div>
<div id="cal-tip"></div>

<script>
const PCOLORS = { codebuddy: '#4f8cff', trae: '#a78bfa', zcode: '#3ecf8e', doubao: '#f0b429' };
function pcolor(p) {
  if (PCOLORS[p]) return PCOLORS[p];
  let h = 0; for (const c of p) h = (h * 31 + c.charCodeAt(0)) % 360;
  return `hsl(${h}, 62%, 62%)`;
}
function esc(s) { return String(s ?? '').replace(/[&<>"']/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function fmtMs(ms) { return ms >= 1000 ? (ms/1000).toFixed(ms >= 10000 ? 0 : 1) + ' s' : ms + ' ms'; }
function fmtUptime(s) {
  if (s < 90) return s + ' 秒';
  if (s < 5400) return Math.round(s/60) + ' 分钟';
  if (s < 172800) return (s/3600).toFixed(1) + ' 小时';
  return Math.round(s/86400) + ' 天';
}
function fmtNum(n) {
  if (n == null) return '—';
  const x = Number(n);
  if (!isFinite(x)) return String(n);
  return x >= 100 ? Math.round(x).toLocaleString() : String(Math.round(x * 10) / 10);
}
function fmtTime(ts) {
  if (!ts) return '—';
  const d = new Date(ts * 1000);
  const pad = n => String(n).padStart(2, '0');
  return `${pad(d.getMonth()+1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}
function toast(msg, isErr) {
  const el = document.getElementById('toast');
  el.textContent = msg; el.className = isErr ? 'err' : ''; el.style.display = 'block';
  clearTimeout(el._t); el._t = setTimeout(() => el.style.display = 'none', 2600);
}
async function api(path, opts) {
  const r = await fetch(path, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error((body.error && body.error.message) || body.detail || (r.status + ' error'));
  return body;
}

let OVERVIEW = null, MODELS = null, STATS = null, BENEFITS = null;

async function loadAll(manual) {
  try {
    const [ov, md, st, bn] = await Promise.all([
      api('/ui/api/overview'), api('/ui/api/models'), api('/ui/api/stats'), api('/ui/api/benefits')]);
    OVERVIEW = ov; MODELS = md; STATS = st; BENEFITS = bn;
    render(); if (manual) toast('已刷新');
  } catch (e) { toast('加载失败: ' + e.message, true); }
}

function render() {
  // 状态条
  const dot = document.getElementById('status-dot');
  const ok = OVERVIEW.authenticated !== false;
  dot.className = 'dot' + (ok ? '' : ' off');
  document.getElementById('status-text').textContent =
    `运行中 · 已运行 ${fmtUptime(OVERVIEW.uptime_seconds)}` + (ok ? '' : ' · codebuddy 未登录');
  document.getElementById('listen-hint').textContent =
    '默认模型: ' + (OVERVIEW.default_model.raw ? OVERVIEW.default_model.raw : '(未设置，按请求原样路由)');

  // 概览卡片
  const s = STATS.summary || {};
  const dm = OVERVIEW.default_model;
  document.getElementById('cards').innerHTML = `
    <div class="card" style="cursor:pointer" title="点击去模型页修改。仅在客户端请求未带 model 字段时生效；不影响已指定 model 的请求" onclick="switchTab('models')"><div class="k">默认启用模型 ↗</div>
      <div class="v">${dm.raw ? `<span class="chip">${esc(dm.provider || 'codebuddy')}</span>${esc(dm.model)}` :
        '<small>未设置（不影响正常使用）</small>'}</div></div>
    <div class="card"><div class="k">24h 请求</div><div class="v">${s.total_24h ?? 0}
      <small>错误 ${s.errors_24h ?? 0}</small></div></div>
    <div class="card"><div class="k">24h 平均耗时</div><div class="v">${s.avg_ms_24h ? fmtMs(s.avg_ms_24h) : '—'}</div></div>
    <div class="card"><div class="k">兜底通道</div><div class="v"><span class="chip">${esc(OVERVIEW.default_provider)}</span><small>未命中模型时</small></div></div>`;

  // 页签角标：模型总数 / 打卡状态
  const totalModels = (MODELS.groups || []).reduce((n, g) => n + g.models.length, 0);
  document.getElementById('cnt-models').textContent = totalModels || '';
  const ck = (BENEFITS.providers || []).filter(p => p.checkin.supported);
  const cb = document.getElementById('cnt-benefits');
  if (ck.length) {
    const pending = ck.filter(p => !p.checkin.done_today && !p.checkin.inactive && !p.checkin.error);
    if (!pending.length) { cb.textContent = '✓ 已签'; cb.className = 'cnt ok'; }
    else { cb.textContent = `未签 ${pending.length}`; cb.className = 'cnt warn'; }
  } else { cb.textContent = ''; }

  renderDaily(); renderModelBars(); renderBenefits(); renderGroups(); renderRecent();
}

// 吸顶偏移：应用顶栏 + 通道条高度（动态测量，写进 CSS 变量供 sticky 使用）
function refreshStickyVars() {
  const tb = document.querySelector('.topbar');
  if (tb) document.documentElement.style.setProperty('--top-h', tb.offsetHeight + 'px');
  const gh = document.querySelector('#page-models .group-head');
  // 模型页未激活时 display:none 测不出高度，保持上次值即可（切页签时会再刷新）
  if (gh && gh.offsetHeight > 0)
    document.documentElement.style.setProperty('--grp-h', gh.offsetHeight + 'px');
}
window.addEventListener('resize', refreshStickyVars);

// ---- 页签 ----
const TABS = ['overview', 'models', 'benefits'];
function switchTab(name) {
  if (!TABS.includes(name)) name = 'overview';
  for (const b of document.querySelectorAll('.tab'))
    b.classList.toggle('active', b.dataset.tab === name);
  for (const s of document.querySelectorAll('.page'))
    s.classList.toggle('active', s.id === 'page-' + name);
  history.replaceState(null, '', '#' + name);
  try { localStorage.setItem('bp-tab', name); } catch (e) {}
  refreshStickyVars();
}
document.querySelectorAll('.tab').forEach(
  b => b.addEventListener('click', () => switchTab(b.dataset.tab)));

// ---- 打卡日历 + 自动打卡 + 各通道额度 ----
function renderBenefits() {
  if (!BENEFITS) return;
  document.getElementById('auto-checkin').checked = !!BENEFITS.auto_checkin;
  document.getElementById('checkin-time').value = BENEFITS.checkin_time || '09:30';
  const enabled = BENEFITS.checkin_enabled_providers || [];
  document.getElementById('checkin-hint').textContent =
    enabled.length ? '每天自动为: ' + enabled.join(', ') : '暂无支持打卡的通道';

  const cal = document.getElementById('cal');
  const days = BENEFITS.calendar || [];
  window._calTips = {};
  const DOW = ['一', '二', '三', '四', '五', '六', '日'];
  const DOW_FULL = ['周一', '周二', '周三', '周四', '周五', '周六', '周日'];
  const now = new Date();
  const pad2 = n => String(n).padStart(2, '0');
  const todayStr = `${now.getFullYear()}-${pad2(now.getMonth() + 1)}-${pad2(now.getDate())}`;
  // 起点补齐到周一，让列与星期对齐
  const first = new Date(now); first.setDate(now.getDate() - (days.length - 1 || 34));
  const padN = (first.getDay() + 6) % 7;
  const cellCount = Math.ceil((days.length + padN) / 7) * 7;
  let html = DOW.map(w => `<div class="dow">${w}</div>`).join('');
  for (let i = 0; i < cellCount; i++) {
    if (i < padN) { html += '<div class="day blank"></div>'; continue; }
    const d = days[i - padN];
    if (!d) { html += '<div class="day blank"></div>'; continue; }
    const hit = (d.providers || []).length > 0;
    html += `<div class="day${hit ? ' hit' : ''}${d.date === todayStr ? ' today' : ''}" data-date="${d.date}">${Number(d.date.slice(8))}</div>`;
  }
  cal.innerHTML = html;
  // 悬停 tooltip 数据：日期 + 星期 + 各通道当日领取
  const dowOf = ds => DOW_FULL[(new Date(ds + 'T12:00:00').getDay() + 6) % 7];
  days.forEach(d => {
    const lines = (d.providers || []).map(pid => {
      const credit = (d.credits || {})[pid];
      return `<div class="t-p"><i style="background:${pcolor(pid)}"></i>${esc(pid)}${credit != null ? ` <b style="color:var(--ok)">+${esc(credit)}</b>` : ''}</div>`;
    });
    window._calTips[d.date] =
      `<div class="t-date">${d.date} ${dowOf(d.date)}${d.date === todayStr ? ' · 今天' : ''}</div>` +
      (lines.length ? lines.join('') : '<div class="t-p muted">当天未打卡</div>');
  });

  const rows = (BENEFITS.providers || []).filter(p => p.checkin.supported);
  document.getElementById('checkin-rows').innerHTML = rows.length ? rows.map(p => {
    const c = p.checkin;
    const bits = [];
    if (c.streak_days >= 2) bits.push(`连续 ${c.streak_days} 天`);
    if (c.daily_credit > 0) bits.push(`每日 +${c.daily_credit}`);
    const extra = bits.length ? ` · ${bits.join(' · ')}` : '';
    const st = c.error ? `<span class="tag bad" title="${esc(c.error)}">状态未知</span>`
      : c.inactive ? '<span class="tag">今日无签到活动</span>'
      : c.done_today ? `<span class="tag ok">已签到${extra}</span>`
      : `<span class="tag">未签到${extra}</span>`;
    const act = c.activity_name ? `<span class="tag ok" title="签到活动档期">${esc(c.activity_name)}</span>` : '';
    return `<div class="benefit-row">
      <span class="mono">${esc(p.id)}</span>${act}${st}<span class="grow"></span>
      <button class="primary" ${(c.done_today || c.inactive) ? 'disabled' : ''} onclick="claimNow('${esc(p.id)}')">立即打卡</button>
    </div>`;
  }).join('') : '<div class="empty" style="padding:14px 0">当前没有支持打卡的通道</div>';

  const qps = (BENEFITS.providers || []).filter(p => p.quota.supported);
  document.getElementById('quota-list').innerHTML = qps.length ? qps.map(p => {
    const q = p.quota;
    const items = (q.items || []).map(it => {
      let pct = it.percent;
      if (pct == null && it.used != null && it.total) pct = Math.round(it.used / it.total * 100);
      const hasBar = pct != null;
      pct = Math.max(0, Math.min(100, Number(pct) || 0));
      const color = pct >= 85 ? 'var(--err)' : pct >= 60 ? 'var(--warn)' : 'var(--ok)';
      let rem = it.remaining;
      if (rem == null && it.used != null && it.total != null) rem = it.total - it.used;
      const hasNums = rem != null || it.total != null;
      const reset = it.reset_ts ? ` · ${new Date(it.reset_ts * 1000).toLocaleString('zh-CN', {month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit'})} 重置` : '';
      const nums = hasNums
        ? `<span class="mono"><span style="color:var(--text)">剩 ${fmtNum(rem)}</span> / ${fmtNum(it.total)} · 已用 ${pct}%</span>`
        : '<span class="mono"></span>';
      const bar = hasBar ? `<div class="qbar"><div style="width:${Math.max(2, pct)}%;background:${color}"></div></div>` : '';
      return `<div class="qitem"><div class="qhead"><span>${esc(it.label)}${reset}</span>${nums}</div>${bar}</div>`;
    }).join('');
    return `<div class="chart-card" style="margin-bottom:12px">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">
        <span style="font-weight:600">${esc(p.name)}</span>
        ${q.level ? `<span class="tag">${esc(q.level)}</span>` : ''}
      </div>${items || '<div class="empty" style="padding:12px 0">无额度数据</div>'}</div>`;
  }).join('') : '<div class="chart-card"><div class="empty" style="padding:14px 0">当前通道均不支持额度查询</div></div>';
}

async function claimNow(pid) {
  try {
    const r = await api('/ui/api/checkin', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({provider: pid})});
    toast(r.ok ? `✓ ${pid} 打卡成功` : `打卡失败: ${r.message || r.error || '未知原因'}`, !r.ok);
    loadAll();
  } catch (e) { toast('打卡失败: ' + e.message, true); }
}

async function saveCheckinSettings() {
  try {
    await api('/ui/api/settings', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        auto_checkin: document.getElementById('auto-checkin').checked,
        checkin_time: document.getElementById('checkin-time').value})});
    toast('自动打卡设置已保存');
    loadAll();
  } catch (e) { toast('保存失败: ' + e.message, true); }
}

// ---- 近14天堆叠柱状图（纯 SVG） ----
function renderDaily() {
  const daily = STATS.daily || [];
  const W = 520, H = 190, padL = 34, padB = 22, padT = 10;
  const max = Math.max(1, ...daily.map(d => d.total));
  const bw = Math.min(34, (W - padL - 8) / daily.length - 6);
  const providers = [...new Set(daily.flatMap(d => Object.keys(d.by_provider)))];
  let svg = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;height:auto">`;
  for (let i = 1; i <= 4; i++) {
    const y = padT + (H - padB - padT) * i / 4;
    const v = max * (4 - i) / 4;
    const lbl = (Number.isInteger(v) && v < 10) || v >= 10 ? Math.round(v) : v.toFixed(1);
    svg += `<line x1="${padL}" y1="${y}" x2="${W-4}" y2="${y}" stroke="#262f3c" stroke-width="1"/>` +
           `<text x="${padL-6}" y="${y+4}" fill="#8b96a5" font-size="10" text-anchor="end">${lbl}</text>`;
  }
  daily.forEach((d, i) => {
    const x = padL + i * ((W - padL - 8) / daily.length) + 3;
    let y = H - padB;
    const segH = n => (H - padB - padT) * n / max;
    for (const p of providers) {
      const n = d.by_provider[p] || 0; if (!n) continue;
      const h = segH(n);
      y -= h;
      svg += `<rect x="${x}" y="${y}" width="${bw}" height="${h}" fill="${pcolor(p)}" rx="2"><title>${d.date} · ${p}: ${n} 次</title></rect>`;
    }
    if (d.total) svg += `<text x="${x + bw/2}" y="${H - 6}" fill="#8b96a5" font-size="9.5" text-anchor="middle">${d.date.slice(5)}</text>`;
  });
  svg += '</svg>';
  document.getElementById('chart-daily').innerHTML = svg;
  document.getElementById('legend-daily').innerHTML =
    providers.map(p => `<span><i style="background:${pcolor(p)}"></i>${esc(p)}</span>`).join('');
}

// ---- 模型请求 Top10 横向条 ----
function renderModelBars() {
  const models = (STATS.models || []).slice(0, 10);
  const el = document.getElementById('chart-models');
  if (!models.length) { el.innerHTML = '<div class="empty">暂无请求数据</div>'; return; }
  const max = Math.max(1, ...models.map(m => m.count));
  el.innerHTML = models.map(m => `
    <div style="display:flex;align-items:center;gap:10px;margin:7px 0">
      <div class="mono" style="width:190px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${esc(m.provider + '/' + m.model)}">
        <i style="display:inline-block;width:8px;height:8px;border-radius:2px;background:${pcolor(m.provider)};margin-right:6px"></i>${esc(m.model)}
      </div>
      <div class="bar-wrap" style="flex:1"><div class="bar" style="width:${Math.max(3, m.count/max*100)}%;background:${pcolor(m.provider)}"></div></div>
      <div class="num muted" style="width:120px;font-size:12px">${m.count} 次 · ${m.avg_ms ? fmtMs(m.avg_ms) : '—'}</div>
    </div>`).join('');
}

// ---- 模型分组表 ----
// 分组折叠状态（localStorage 持久化；30s 自动刷新重渲染后仍保持）
let COLLAPSED = new Set();
try { COLLAPSED = new Set(JSON.parse(localStorage.getItem('bp-collapsed') || '[]')); } catch (e) {}
function persistCollapsed() {
  try { localStorage.setItem('bp-collapsed', JSON.stringify([...COLLAPSED])); } catch (e) {}
}
function toggleGroup(gid) {
  if (COLLAPSED.has(gid)) COLLAPSED.delete(gid); else COLLAPSED.add(gid);
  persistCollapsed();
  const g = document.querySelector(`.group[data-gid="${CSS.escape(gid)}"]`);
  if (g) g.classList.toggle('collapsed', COLLAPSED.has(gid));
}
function setAllGroups(collapse) {
  COLLAPSED = collapse ? new Set((MODELS.groups || []).map(g => g.id)) : new Set();
  persistCollapsed();
  document.querySelectorAll('#groups .group').forEach(g =>
    g.classList.toggle('collapsed', COLLAPSED.has(g.dataset.gid)));
}
// 点击通道条或列头（表头）任意处 → 折叠/展开该组
document.getElementById('groups').addEventListener('click', e => {
  const g = e.target.closest('.group');
  if (!g) return;
  if (e.target.closest('.group-head') || e.target.closest('thead th')) {
    if (!e.target.closest('button')) toggleGroup(g.dataset.gid);
  }
});

// 打卡日历悬停 tooltip（自绘，即时显示：日期 + 星期 + 各通道当日领取）
const calTip = document.getElementById('cal-tip');
document.getElementById('cal').addEventListener('mousemove', e => {
  const cell = e.target.closest('.day');
  const tip = cell && !cell.classList.contains('blank')
    && window._calTips && window._calTips[cell.dataset.date];
  if (!tip) { calTip.style.display = 'none'; return; }
  calTip.innerHTML = tip;
  calTip.style.display = 'block';
  const r = cell.getBoundingClientRect();
  let x = r.left + r.width / 2 - calTip.offsetWidth / 2;
  x = Math.max(8, Math.min(x, window.innerWidth - calTip.offsetWidth - 8));
  let y = r.top - calTip.offsetHeight - 8;
  if (y < 8) y = r.bottom + 8;
  calTip.style.left = x + 'px';
  calTip.style.top = y + 'px';
});
document.getElementById('cal').addEventListener('mouseleave', () => {
  calTip.style.display = 'none';
});

function renderGroups() {
  const el = document.getElementById('groups');
  const groups = (MODELS.groups || []);
  const totalModels = groups.reduce((n, g) => n + g.models.length, 0);
  const dm = MODELS.default_model || '';
  document.getElementById('model-summary').textContent =
    `${groups.length} 个通道 · ${totalModels} 个模型` + (dm ? ` · 默认 ${dm}` : '');
  if (!groups.length) { el.innerHTML = '<div class="empty">没有可用通道</div>'; return; }
  el.innerHTML = groups.map(g => {
    const h = g.health || {};
    const okFlag = (g.id === 'codebuddy') ? (h.authenticated && h.token_valid !== false)
                 : ('configured' in h ? h.configured !== false : (h.ready !== false && h.started !== false));
    const healthTxt = g.id === 'codebuddy' ? (okFlag ? '已登录' : '未登录')
                    : h.base_url ? esc(h.base_url.replace(/^https?:\/\//, '')) : (okFlag ? '正常' : '未就绪');
    const rows = g.models.map(m => {
      const st = m.stats || {};
      return `<tr>
        <td><span class="mono">${esc(m.id)}</span>
            ${m.is_default ? '<span class="badge-default">✦ 默认</span>' : ''}</td>
        <td class="muted">${esc(m.name || '')}</td>
        <td>${m.credits ? `<span class="tag">${esc(m.credits)}</span>` : ''}${m.tier ? `<span class="tag">${esc(m.tier)}</span>` : ''}${m.reasoning ? '<span class="tag">reasoning</span>' : ''}</td>
        <td class="num mono muted">${st.count || '—'}</td>
        <td class="num mono muted">${st.count ? fmtMs(st.avg_ms) : '—'}</td>
        <td class="num" style="white-space:nowrap">
            ${m.is_default ? '' : `<button title="客户端请求未带 model 字段时，自动改用此模型（不影响已指定 model 的请求）" onclick="setDefault('${esc(g.id)}','${esc(m.id)}')">设为默认</button>`}
            <button class="primary" onclick="runTest('${esc(g.id)}','${esc(m.id)}')">测试</button>
        </td></tr>`;
    }).join('');
    return `<div class="group${COLLAPSED.has(g.id) ? ' collapsed' : ''}" data-gid="${esc(g.id)}">
      <div class="group-head" title="点击折叠/展开">
        <span class="dot" style="background:${okFlag ? 'var(--ok)' : 'var(--err)'}"></span>
        <span class="name">${esc(g.name)}</span>
        <span class="tag">${g.id}</span>
        <span class="tag ${okFlag ? 'ok' : 'bad'}">${healthTxt}</span>
        <span class="spacer"></span>
        <span class="muted" style="font-size:12px">${g.models.length} 个模型</span>
        <span class="chev">▾</span>
      </div>
      <table>
        <thead><tr><th>模型 ID</th><th>名称</th><th>标签</th><th class="num">14d 请求</th><th class="num">平均耗时</th><th class="num">操作</th></tr></thead>
        <tbody>${rows}</tbody>
      </table></div>`;
  }).join('');
  refreshStickyVars();
}

function renderRecent() {
  const rows = STATS.recent || [];
  const el = document.getElementById('recent');
  if (!rows.length) { el.innerHTML = '<tr><td colspan="7" class="empty">还没有请求记录</td></tr>'; return; }
  el.innerHTML = rows.slice(0, 50).map(r => {
    const bad = (r.status >= 400 || r.error);
    return `<tr>
      <td class="mono muted">${fmtTime(r.ts)}</td>
      <td><i style="display:inline-block;width:8px;height:8px;border-radius:2px;background:${pcolor(r.provider)};margin-right:6px"></i>${esc(r.provider)}</td>
      <td class="mono">${esc(r.model)}</td>
      <td class="muted">${esc(r.protocol || '')}${r.stream ? ' ⚡' : ''}</td>
      <td class="num mono">${r.duration_ms ? fmtMs(r.duration_ms) : '—'}</td>
      <td class="num mono muted">${(r.prompt_tokens || r.completion_tokens) ? (r.prompt_tokens + '+' + r.completion_tokens) : '—'}</td>
      <td>${bad ? `<span class="tag bad" title="${esc(r.error || '')}">${r.status || 'ERR'}</span>` : '<span class="tag ok">OK</span>'}</td>
    </tr>`;
  }).join('');
}

// ---- 操作 ----
async function setDefault(provider, model) {
  try {
    await api('/ui/api/settings', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ default_model: provider + '/' + model })});
    toast(`默认模型已设为 ${provider}/${model}`);
    loadAll();
  } catch (e) { toast('设置失败: ' + e.message, true); }
}

async function runTest(provider, model) {
  const overlay = document.getElementById('overlay');
  document.getElementById('modal-title').textContent = `测试 ${provider}/${model}`;
  document.getElementById('modal-body').innerHTML = '<span class="spin"></span>发送 "hi" 到上游，等待回复…（最长 120s）';
  overlay.classList.add('show');
  try {
    const r = await api('/ui/api/test', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ provider, model })});
    const u = r.usage || {};
    document.getElementById('modal-body').innerHTML = r.ok ? `
      <div class="kv">
        <span class="k">结果</span><span style="color:var(--ok)">✓ 成功</span>
        <span class="k">耗时</span><span class="mono">${fmtMs(r.latency_ms)}</span>
        <span class="k">回复模型</span><span class="mono">${esc(r.model || '')}</span>
        <span class="k">Tokens</span><span class="mono">${u.prompt_tokens ?? '—'} + ${u.completion_tokens ?? '—'}</span>
        <span class="k">finish</span><span class="mono">${esc(r.finish_reason || '—')}</span>
      </div>
      <pre class="view">${esc(r.content)}</pre>` : `
      <div class="kv">
        <span class="k">结果</span><span style="color:var(--err)">✗ 失败</span>
        <span class="k">耗时</span><span class="mono">${fmtMs(r.latency_ms)}</span>
        <span class="k">状态</span><span class="mono">${r.status || '—'}</span>
      </div>
      <pre class="view" style="color:#ffb3b3">${esc(r.error || '未知错误')}</pre>`;
    loadAll();
  } catch (e) {
    document.getElementById('modal-body').innerHTML =
      `<pre class="view" style="color:#ffb3b3">${esc(e.message)}</pre>`;
  }
}

function closeModal() { document.getElementById('overlay').classList.remove('show'); }
document.getElementById('overlay').addEventListener('click', e => {
  if (e.target === e.currentTarget) closeModal();
});
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });

let initTab = location.hash.replace('#', '');
try { initTab = initTab || localStorage.getItem('bp-tab') || ''; } catch (e) {}
switchTab(initTab);
loadAll();
setInterval(loadAll, 30000);
</script>
</body>
</html>
"""
