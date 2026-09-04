#!/usr/bin/env python3
"""扫描 ethan 会话库，捞出「工具调用语法泄漏进正文」的消息。

下次再遇到"回复里出现 {"name": ... / <tool_call / seed:tool_call 之类"
的泄漏，不用再翻聊天记录：跑

    python3 collect_badcases.py            # 扫描并列出
    python3 collect_badcases.py --save     # 顺带落盘 fixtures/badcases/

签名判定：assistant 消息正文里含调用语法特征、但 tool_calls 字段为空
（即泄漏而非正常调用）。输出消息 id / 会话 / 模型 / 特征，可直接定位到
http://localhost:8900/chat/<session_id> 复核；--save 后把 payload 内联进
test_badcase_*.py 或用其生成回归断言。

只读打开 db，不写任何数据。
"""
import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

DB = Path.home() / ".ethan/db/sessions.db"
OUT = Path(__file__).resolve().parent / "fixtures" / "badcases"

# 泄漏签名：正文含调用语法特征（tool_calls 为空 = 没走正常调用通道）
SIGS = {
    "bare_json": re.compile(r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"'),
    "loose_kv": re.compile(r'name\s*:\s*"[A-Za-z_][\w.-]*"\s*,?\s*arguments\s*:'),
    "tagged": re.compile(r"<\s*/?\s*tool_call[^>]*>"),
    "attr_tag": re.compile(r'<tool_call\s+name="'),
    "seed": re.compile(r"<seed:tool_call>"),
    "seed_fn": re.compile(r"<function\s+name="),
    "fn_expr": re.compile(r'\b\w+\s*\(\s*\w+\s*=\s*"'),
    "arg_debris": re.compile(r"</?\s*arg_(?:key|value)"),
    "tool_callback": re.compile(r"<tool_callback"),
}
# 这些特征常见于正常正文，单独出现不算泄漏
WEAK = {"fn_expr", "arg_debris"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=DB)
    ap.add_argument("--save", action="store_true", help="落盘 payload 到 fixtures/badcases/")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--session", help="只看指定会话")
    args = ap.parse_args()

    if not args.db.exists():
        print(f"db 不存在: {args.db}", file=sys.stderr)
        return 2
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    q = ("SELECT id, session_id, model, content, datetime(created_at,'unixepoch','localtime') "
         "FROM messages WHERE role='assistant' AND (tool_calls IS NULL OR tool_calls='') "
         "AND status='completed'")
    if args.session:
        q += f" AND session_id='{args.session}'"
    q += " ORDER BY id DESC"
    hits = []
    for mid, sid, model, content, ts in con.execute(q):
        if not content:
            continue
        found = [name for name, pat in SIGS.items() if pat.search(content)]
        if not found or set(found) <= WEAK:
            continue
        hits.append((mid, sid, model, ts, found, content))
        if len(hits) >= args.limit:
            break

    if not hits:
        print("未发现泄漏签名（好现象）。")
        return 0
    print(f"发现 {len(hits)} 条疑似泄漏：\n")
    for mid, sid, model, ts, found, content in hits:
        print(f"msg {mid}  {sid}  {model}  {ts}")
        print(f"  签名: {','.join(found)}")
        print(f"  复核: http://localhost:8900/chat/{sid}")
        i = min((content.find(p) for p in (
            '{"name"', "<tool_call", "<seed:tool_call", "name: \"") if p != -1
            and content.find(p) != -1), default=0)
        print(f"  片段: {content[max(0, i - 40):i + 100]!r}\n")
    if args.save:
        OUT.mkdir(parents=True, exist_ok=True)
        for mid, sid, model, ts, found, content in hits:
            i = min((content.find(p) for p in (
                '{"name"', "<tool_call", "<seed:tool_call", "name: \"") if p != -1
                and content.find(p) != -1), default=0)
            payload = content[max(0, i - 60):]
            meta = {"id": mid, "session": sid, "model": model, "ts": ts,
                    "signatures": found}
            f = OUT / f"msg_{mid}.json"
            f.write_text(json.dumps(
                {"meta": meta, "payload": payload}, ensure_ascii=False, indent=2))
            print(f"saved -> {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
