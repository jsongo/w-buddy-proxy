"""打卡 / 额度：打卡历史落盘 + 自动打卡后台任务（/ui 管理页消费）。

能力约定见 ``providers.BaseProvider``：provider 以 ``supports_checkin = True``
声明支持打卡，``checkin_status()/checkin_claim()/quota()`` 返回 None 表示
不支持对应能力。所有上游调用都是同步的，经 ``asyncio.to_thread`` 执行。

打卡历史逐行落盘 ``logs/checkin.jsonl``（手动 / 自动都走同一条路），管理页
的「打卡日历」由此渲染，自动打卡任务以此判断当天是否已完成。
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import datetime, timedelta
from typing import Any, Callable

from buddy_proxy import settings as settings_mod

# 自动打卡巡检周期；失败重试间隔
CHECK_INTERVAL_S = 600
RETRY_THROTTLE_S = 1800
# 状态/额度缓存 TTL：管理页 30s 自动刷新，不能每次都打上游
SNAPSHOT_TTL_S = 300
DEFAULT_CHECKIN_TIME = "09:30"


def _today() -> str:
    return time.strftime("%Y-%m-%d")


def read_checkin_settings() -> dict[str, Any]:
    saved = settings_mod.load_settings()
    raw_time = str(saved.get("checkin_time") or DEFAULT_CHECKIN_TIME)
    return {
        "auto_checkin": bool(saved.get("auto_checkin", True)),
        "checkin_time": raw_time if re.fullmatch(r"\d{1,2}:\d{2}", raw_time) else DEFAULT_CHECKIN_TIME,
    }


class CheckinHistory:
    """append-only 打卡历史（JSONL）。"""

    def __init__(self, path):
        self.path = path

    def append(self, provider: str, ok: bool, message: str = "",
               credits: Any = None, date: str | None = None) -> None:
        rec = {
            "ts": round(time.time(), 3),
            "date": date or _today(),
            "provider": provider,
            "ok": bool(ok),
            "message": (message or "")[:200],
            "credits": credits,
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass  # 历史落盘失败不影响打卡本身

    def entries(self) -> list[dict[str, Any]]:
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return []
        out = []
        for line in lines:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                out.append(rec)
        return out

    def ok_dates_by_provider(self) -> dict[str, set[str]]:
        """provider -> 打卡成功的日期集合（含上游返回"已签到"的记录）。"""
        dates: dict[str, set[str]] = {}
        for rec in self.entries():
            if rec.get("ok"):
                dates.setdefault(rec.get("provider", ""), set()).add(rec.get("date", ""))
        return dates

    def calendar(self, days: int = 35) -> list[dict[str, Any]]:
        """最近 N 天的打卡日历，旧日期在前。

        每天一项：``{"date", "providers": [id...], "credits": {id: 积分}}``，
        credits 是当天实际领取的签到积分（上游签到历史合并进来的日子没有
        本地领取记录，只有 provider 名）。
        """
        by_date: dict[str, dict[str, Any]] = {}
        for rec in self.entries():
            if not (rec.get("ok") and rec.get("date")):
                continue
            info = by_date.setdefault(rec["date"], {"providers": [], "credits": {}})
            pid = rec.get("provider", "")
            if pid not in info["providers"]:
                info["providers"].append(pid)
            if rec.get("credits") is not None:
                info["credits"][pid] = rec["credits"]
        today = datetime.now().date()
        out = []
        for offset in range(days - 1, -1, -1):
            date = (today - timedelta(days=offset)).isoformat()
            info = by_date.get(date, {"providers": [], "credits": {}})
            out.append({
                "date": date,
                "providers": sorted(info["providers"]),
                "credits": info["credits"],
            })
        return out


class BenefitsManager:
    """打卡状态聚合 + 自动打卡后台循环。挂在 ``ProxyState.benefits`` 上。"""

    def __init__(self, history_path, state: Any):
        self.history = CheckinHistory(history_path)
        self._state = state
        self._task: asyncio.Task | None = None
        self._last_attempt: dict[str, float] = {}
        self._startup_seen: set[str] = set()
        self._cache: dict[str, tuple[float, Any]] = {}

    # ------------------------------------------------------------------
    # provider 能力
    # ------------------------------------------------------------------

    def _providers(self) -> dict[str, Any]:
        """已启用 provider + 默认 codebuddy 通道（打卡/额度能力均不支持，UI 显示占位）。"""
        providers = dict(getattr(self._state, "providers", {}) or {})
        try:
            from buddy_proxy.codebuddy_provider import _default_codebuddy
            providers.setdefault("codebuddy", _default_codebuddy)
        except Exception:
            pass
        return providers

    def checkin_providers(self) -> dict[str, Any]:
        return {
            pid: p for pid, p in self._providers().items()
            if getattr(p, "supports_checkin", False)
        }

    # ------------------------------------------------------------------
    # 快照（GET /ui/api/benefits）
    # ------------------------------------------------------------------

    async def snapshot(self) -> dict[str, Any]:
        checkin_cfg = read_checkin_settings()
        done = self.history.ok_dates_by_provider()
        today = _today()

        provider_entries = []
        for pid, p in self._providers().items():
            supports_checkin = bool(getattr(p, "supports_checkin", False))
            entry: dict[str, Any] = {
                "id": pid,
                "name": getattr(p, "name", pid),
                "checkin": {"supported": supports_checkin},
                "quota": {"supported": False},
            }
            if supports_checkin:
                status = await self._cached(f"checkin:{pid}", p.checkin_status)
                done_today = today in done.get(pid, set()) or bool(
                    status and status.get("checked_in"))
                entry["checkin"].update({
                    "done_today": done_today,
                    **(status or {}),
                })
            quota = await self._cached(f"quota:{pid}", p.quota)
            if quota is not None:
                entry["quota"] = {"supported": True, **quota}
            provider_entries.append(entry)

        # 上游直接给出的签到历史（如 CodeBuddy checkin_dates）合并进日历，
        # 这样代理没记录过的历史打卡天也能展示（无本地领取积分记录）
        day_map: dict[str, dict[str, Any]] = {}
        day_order: list[str] = []
        for d in self.history.calendar(35):
            day_map[d["date"]] = {
                "providers": set(d["providers"]),
                "credits": dict(d.get("credits") or {}),
            }
            day_order.append(d["date"])
        for entry in provider_entries:
            pid = entry["id"]
            for ds in entry["checkin"].get("checkin_dates") or []:
                if ds in day_map and pid not in day_map[ds]["providers"]:
                    day_map[ds]["providers"].add(pid)
        calendar_out = [
            {"date": d,
             "providers": sorted(day_map[d]["providers"]),
             "credits": day_map[d]["credits"]}
            for d in day_order
        ]

        return {
            "providers": provider_entries,
            "calendar": calendar_out,
            "auto_checkin": checkin_cfg["auto_checkin"],
            "checkin_time": checkin_cfg["checkin_time"],
            "checkin_enabled_providers": sorted(self.checkin_providers()),
        }

    async def _cached(self, key: str, fn: Callable, *args):
        """TTL 缓存的 to_thread 调用；上游抛错时返回错误结构而不是 500。"""
        now = time.time()
        cached = self._cache.get(key)
        if cached and now - cached[0] < SNAPSHOT_TTL_S:
            return cached[1]
        try:
            data = await asyncio.to_thread(fn, *args)
        except Exception as exc:
            data = {"error": str(exc)[:300]}
        self._cache[key] = (now, data)
        return data

    # ------------------------------------------------------------------
    # 手动 / 自动打卡
    # ------------------------------------------------------------------

    async def claim_now(self, provider_id: str) -> dict[str, Any]:
        provider = self.checkin_providers().get(provider_id)
        if provider is None:
            return {"ok": False, "error": f"provider {provider_id} 不支持打卡"}
        result = await self._claim_and_record(provider_id, provider)
        self._cache.pop(f"checkin:{provider_id}", None)
        return result

    async def _claim_and_record(self, provider_id: str, provider: Any) -> dict[str, Any]:
        try:
            status = await asyncio.to_thread(provider.checkin_claim)
        except Exception as exc:
            message = str(exc)[:300]
            # 上游返回"已签到"类提示视为成功（幂等补记录）
            already = ("已签" in message) or ("already" in message.lower())
            self.history.append(provider_id, ok=already, message=message)
            return {"ok": already, "already": already, "message": message}
        self.history.append(provider_id, ok=True,
                            message=status.get("message", ""),
                            credits=status.get("extra_credits", status.get("credits")))
        return {"ok": True, **(status or {})}

    # ------------------------------------------------------------------
    # 后台自动打卡循环
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._run())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def _run(self) -> None:
        while True:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                pass  # 自动打卡异常不影响代理转发
            await asyncio.sleep(CHECK_INTERVAL_S)

    async def _tick(self) -> None:
        cfg = read_checkin_settings()
        if not cfg["auto_checkin"]:
            return
        hh, mm = (int(x) for x in cfg["checkin_time"].split(":"))
        now = datetime.now()
        due_today = (now.hour, now.minute) >= (hh, mm)
        done = self.history.ok_dates_by_provider()
        for pid, provider in self.checkin_providers().items():
            if _today() in done.get(pid, set()):
                continue
            first_run = pid not in self._startup_seen
            self._startup_seen.add(pid)
            # 启动后首轮视为补签窗口（不管是否到点）；之后只在到达设定时刻后打
            if not due_today and not first_run:
                continue
            if time.time() - self._last_attempt.get(pid, 0.0) < RETRY_THROTTLE_S:
                continue
            self._last_attempt[pid] = time.time()
            # 先查状态再决定是否领：避免对「已签到/当天无活动」的上游反复打 claim
            try:
                status = await asyncio.to_thread(provider.checkin_status)
            except Exception:
                continue
            if not isinstance(status, dict):
                continue
            if status.get("checked_in"):
                # 上游已签到（如网页/客户端手动签过）→ 补记历史，当天不再重试
                self.history.append(pid, ok=True,
                                    message=status.get("message") or "已签到（上游记录）")
                continue
            if status.get("claimable") is False:
                continue  # 当天无签到活动（如 CodeBuddy 档期未开），保持轻量轮询
            await self._claim_and_record(pid, provider)
