"""请求指标收集：内存聚合 + JSONL 落盘，供 /ui 管理页按 provider/模型维度聚合。

每次 /v1/chat/completions、/v1/responses、/v1/messages 请求完成（或失败）时，
``codebuddy_provider.forward_chat`` 经由 :meth:`MetricsCollector.record` 记一条：

- 内存里按 ``(日期, provider, model)`` 聚合（请求数 / 错误数 / 耗时 / token），
  另保留最近 N 条明细供「最近请求」表格展示；
- 同时追加一行 JSON 到 ``logs/metrics.jsonl``，重启后回读文件尾部恢复聚合，
  保证图表跨重启不丢历史。
"""

from __future__ import annotations

import collections
import json
import os
import pathlib
import threading
import time
from typing import Any, Optional

# 内存里保留的最近请求明细条数（「最近请求」表格用）
MAX_RECENT = 200
# 启动时回读 metrics.jsonl 的尾部字节数（单条约 200B，约万条请求）
_TAIL_BYTES = 2_000_000
# 聚合只保留最近 N 天，更早的启动回读时丢弃
KEEP_DAYS = 30


def _date_str(ts: float) -> str:
    return time.strftime("%Y-%m-%d", time.localtime(ts))


class MetricsCollector:
    """线程安全的请求指标收集器。"""

    def __init__(self, log_path: Optional[os.PathLike | str] = None):
        self.log_path = pathlib.Path(log_path) if log_path else None
        self._lock = threading.Lock()
        # (date, provider, model) -> 聚合桶
        self._daily: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._recent: collections.deque = collections.deque(maxlen=MAX_RECENT)
        self._load_tail()

    # ------------------------------------------------------------------
    # 记录
    # ------------------------------------------------------------------

    def record(
        self,
        provider: str,
        model: str,
        protocol: str = "",
        status: int = 200,
        duration_ms: int = 0,
        stream: bool = False,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        error: str = "",
        chunk_count: int = 0,
    ) -> None:
        ts = time.time()
        rec = {
            "ts": round(ts, 3),
            "provider": provider or "codebuddy",
            "model": model or "(unknown)",
            "protocol": protocol,
            "status": int(status),
            "duration_ms": int(duration_ms),
            "stream": bool(stream),
            "prompt_tokens": int(prompt_tokens or 0),
            "completion_tokens": int(completion_tokens or 0),
            "chunk_count": int(chunk_count or 0),
            "error": (error or "")[:300],
        }
        with self._lock:
            self._recent.append(rec)
            self._bucket_add(self._bucket(_date_str(ts), rec["provider"], rec["model"]), rec)
        self._append_log(rec)

    def _bucket(self, date: str, provider: str, model: str) -> dict[str, Any]:
        key = (date, provider, model)
        b = self._daily.get(key)
        if b is None:
            b = {
                "count": 0,
                "errors": 0,
                "duration_ms_sum": 0,
                "duration_ms_max": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "last_ts": 0.0,
            }
            self._daily[key] = b
        return b

    @staticmethod
    def _bucket_add(b: dict[str, Any], rec: dict[str, Any]) -> None:
        is_error = rec["status"] >= 400 or bool(rec["error"])
        b["count"] += 1
        b["errors"] += 1 if is_error else 0
        b["duration_ms_sum"] += rec["duration_ms"]
        b["duration_ms_max"] = max(b["duration_ms_max"], rec["duration_ms"])
        b["prompt_tokens"] += rec["prompt_tokens"]
        b["completion_tokens"] += rec["completion_tokens"]
        b["last_ts"] = max(b["last_ts"], rec.get("ts", 0))

    def _append_log(self, rec: dict[str, Any]) -> None:
        if self.log_path is None:
            return
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass  # 指标落盘失败不影响转发

    # ------------------------------------------------------------------
    # 启动恢复：回读文件尾部
    # ------------------------------------------------------------------

    def _load_tail(self) -> None:
        if self.log_path is None or not self.log_path.exists():
            return
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - _TAIL_BYTES))
                blob = f.read().decode("utf-8", errors="replace")
        except Exception:
            return
        cutoff = time.time() - KEEP_DAYS * 86400
        with self._lock:
            for line in blob.splitlines()[1 if size > _TAIL_BYTES else 0:]:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict) or rec.get("ts", 0) < cutoff:
                    continue
                self._recent.append(rec)
                self._bucket_add(
                    self._bucket(_date_str(rec["ts"]), rec.get("provider", ""), rec.get("model", "")),
                    {
                        "status": rec.get("status", 200),
                        "duration_ms": rec.get("duration_ms", 0),
                        "prompt_tokens": rec.get("prompt_tokens", 0),
                        "completion_tokens": rec.get("completion_tokens", 0),
                        "error": rec.get("error", ""),
                    },
                )

    # ------------------------------------------------------------------
    # 查询（/ui/api/stats）
    # ------------------------------------------------------------------

    def snapshot(self, days: int = 14) -> dict[str, Any]:
        """返回按模型聚合 + 按天序列 + 最近请求明细。"""
        now = time.time()
        with self._lock:
            daily = {k: dict(v) for k, v in self._daily.items()}
            recent = list(self._recent)

        # ---- 按模型聚合（全部保留窗口内） ----
        per_model: dict[tuple[str, str], dict[str, Any]] = {}
        for (date, provider, model), b in daily.items():
            agg = per_model.setdefault(
                (provider, model),
                {
                    "provider": provider,
                    "model": model,
                    "count": 0,
                    "errors": 0,
                    "duration_ms_sum": 0,
                    "duration_ms_max": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "last_ts": 0.0,
                },
            )
            agg["count"] += b["count"]
            agg["errors"] += b["errors"]
            agg["duration_ms_sum"] += b["duration_ms_sum"]
            agg["duration_ms_max"] = max(agg["duration_ms_max"], b["duration_ms_max"])
            agg["prompt_tokens"] += b["prompt_tokens"]
            agg["completion_tokens"] += b["completion_tokens"]
            agg["last_ts"] = max(agg["last_ts"], b["last_ts"])
        models = []
        for agg in per_model.values():
            count = agg["count"]
            agg["avg_ms"] = round(agg.pop("duration_ms_sum") / count) if count else 0
            models.append(agg)
        models.sort(key=lambda m: -m["count"])

        # ---- 按天序列（零填充，图表连续） ----
        day_list = []
        for offset in range(days - 1, -1, -1):
            day_start = now - offset * 86400
            date = _date_str(day_start)
            day_list.append({"date": date, "total": 0, "errors": 0, "by_provider": {}})
        by_date = {d["date"]: d for d in day_list}
        for (date, provider, model), b in daily.items():
            d = by_date.get(date)
            if d is None:
                continue
            d["total"] += b["count"]
            d["errors"] += b["errors"]
            d["by_provider"][provider] = d["by_provider"].get(provider, 0) + b["count"]

        # ---- 24h 汇总 + 最近请求 ----
        # 口径限制：24h 汇总基于 recent deque（上限 MAX_RECENT 条）。24h 内请求
        # 超过上限时会少算——「最近请求」表格同理只展示最近 50 条。对单机自用
        # 够用；如需精确按小时分桶，需要额外状态，暂不做。
        total_24h = errors_24h = 0
        dur_sum_24h = 0
        for rec in recent:
            if rec.get("ts", 0) >= now - 86400:
                total_24h += 1
                dur_sum_24h += rec.get("duration_ms", 0)
                if rec.get("status", 200) >= 400 or rec.get("error"):
                    errors_24h += 1

        return {
            "models": models,
            "daily": day_list,
            "recent": list(reversed(recent[-50:])),
            "summary": {
                "total_24h": total_24h,
                "errors_24h": errors_24h,
                "avg_ms_24h": round(dur_sum_24h / total_24h) if total_24h else 0,
            },
        }
