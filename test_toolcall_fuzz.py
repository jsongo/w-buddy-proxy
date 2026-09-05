"""工具调用解析系统性变异 fuzz + 定向 case 矩阵。

目标：不再「发现一个 badcase 修一个」。对每种合法调用语法做系统性变异
（逐偏移截断 / 删结构字符 / 去引号转义 / 注入协议噪音 / 换行重组），
流式与非流式双路径跑，断言两条不变量：

A. 不泄漏：变异后仍带已知工具名的输入，输出正文里不得出现调用语法
   残骸（{"name" / "arguments" / <tool_call / name: "工具" / 工具名(），
   要么解析成调用、要么被闸门丢弃；
B. 不误伤：正常正文语料（YAML / 配置 JSON / SVG / 代码示例）不产调用、
   不被吞。

确定性：固定随机种子，失败可复现。
"""
import json
import random
import re
import sys

sys.path.insert(0, "src")
from buddy_proxy.trae_provider import (  # noqa: E402
    _parse_tool_calls,
    _StreamToolCallSplitter,
)

O, C = "<tool_call>", "</tool_call>"
KNOWN_LIST = ["shell", "file_read", "file_write", "web_search", "recall_memory"]
KNOWN = frozenset(KNOWN_LIST)
TOOLS_RE = "|".join(KNOWN_LIST)

fails = []
total = 0
_rng = random.Random(20260905)


def check(name, cond, extra=""):
    global total
    total += 1
    if not cond:
        print(f"[FAIL] {name} {extra}")
        fails.append(name)


# ---------------------------------------------------------------- 不变量

_LEAK_RES = [
    re.compile(r'\{\s*"name"'),
    re.compile(r'"arguments"'),
    re.compile(r"<\s*/?\s*tool_call"),
    re.compile(r"</?\s*arg_(?:key|value)"),
    re.compile(rf'name:\s*"(?:{TOOLS_RE})"'),
    re.compile(rf'(?:{TOOLS_RE})\s*\('),
]


def leak_markers(text):
    """text 里的调用语法残骸（不变量 A 的检测目标）。"""
    return [p.pattern for p in _LEAK_RES if p.search(text)]


def run_parse(mutated, chunk=None):
    """跑一条路径，返回 (输出正文, calls)。chunk=None 非流式。"""
    if chunk is None:
        rest, calls = _parse_tool_calls(mutated, KNOWN)
        return rest, calls
    sp = _StreamToolCallSplitter(KNOWN)
    out = []
    for k in range(0, len(mutated), chunk):
        out.append(sp.feed(mutated[k:k + chunk]))
    tail, calls = sp.flush()
    return "".join(out) + tail, calls


def assert_no_leak(label, mutated, chunk=None):
    """不变量 A：仍带已知工具名的输入，输出不得泄漏调用语法。"""
    if not any(t in mutated for t in KNOWN_LIST):
        return
    try:
        content, calls = run_parse(mutated, chunk)
    except Exception as ex:  # noqa: BLE001
        check(f"{label}", False, f"异常 {ex!r} input={mutated[:80]!r}")
        return
    marks = leak_markers(content)
    check(f"{label}", not marks,
          f"泄漏 {marks} content={content[:120]!r} input={mutated[:120]!r}")


def assert_prose_kept(label, prose):
    """不变量 B：正文不产调用、关键内容不被吞。"""
    try:
        for chunk in (None, 3, 17):
            content, calls = run_parse(prose, chunk)
            check(f"{label}[chunk={chunk}]", len(calls) == 0,
                  f"误产调用 {[c['function']['name'] for c in calls]}")
            check(f"{label}[chunk={chunk}]内容", prose.strip()[:40] in content
                  or content.strip()[:40] in prose,
                  f"被吞 content={content[:80]!r}")
    except Exception as ex:  # noqa: BLE001
        check(f"{label}", False, f"异常 {ex!r}")


def assert_valid_call(label, text, want_name, want_args_subset=None, chunks=(1, 3, 7, 17, 64, None)):
    """合法调用：非流式 + 各流式切点都必须解析成功且无泄漏。"""
    for chunk in chunks:
        try:
            content, calls = run_parse(text, chunk)
        except Exception as ex:  # noqa: BLE001
            check(f"{label}[chunk={chunk}]", False, f"异常 {ex!r}")
            continue
        ok = len(calls) == 1 and calls[0]["function"]["name"] == want_name
        extra = f"n={len(calls)}"
        if ok and want_args_subset:
            got = json.loads(calls[0]["function"]["arguments"])
            for k, v in want_args_subset.items():
                if got.get(k) != v:
                    ok = False
                    extra += f" {k}={got.get(k)!r}≠{v!r}"
        check(f"{label}[chunk={chunk}]", ok, extra)
        check(f"{label}[chunk={chunk}]无残骸", not leak_markers(content),
              f"content={content[:80]!r}")


# ---------------------------------------------------------------- 合法语法语料

VALID = {
    "tagged": (
        '我先查一下目录。\n' + O + '\n{"name": "shell", "arguments":'
        ' {"command": "ls -la /tmp", "intent": "查临时目录"}}\n' + C + '\n'),
    "bare": (
        '配置写进文件。\n{"name": "file_write", "arguments":'
        ' {"content": "line1\\nline2 \\"quoted\\"", "path": "/tmp/a.py"}}\n'),
    "bare_flat": '{"name": "web_search", "query": "claude 检测", "max_results": 7}\n',
    "loose_kv": (
        '让我查一下。\nname: "shell"\narguments: {"command": "ps aux | head -5",'
        ' "intent": "查进程"}\n'),
    "loose_kv_inline": 'name: "file_read" arguments: {"path": "/etc/hosts"}',
    "fn_expr": (
        '查一下。\nweb_search(query="claude fingerprint 检测", max_results=7)'
        ' web_search(query="glm 退化")\n'),
    "colon": 'web_searchquery: claude fingerprint 检测方法',
    "attr": ('查一下。\n<tool_call name="shell" command="ls /tmp 2>/dev/null" intent="查" />\n'),
    "tagged_multi": (
        O + '\n{"name": "shell", "arguments": {"command": "pwd"}}\n' + C + "\n"
        + O + '\n{"name": "file_read", "arguments": {"path": "/tmp/x"}}\n' + C + "\n"),
    "tagged_escape": (
        O + '\n{"name": "shell", "arguments": {"command": "echo \\"---\\" && ls"}}\n' + C + "\n"),
    "tagged_unicode": (
        O + '\n{"name": "shell", "arguments": {"command": "echo 中文测试", "note": "\\u4e2d\\u6587"}}\n' + C + "\n"),
    "loose_kv_crlf": 'name: "shell"\r\narguments: {"command": "uptime"}\r\n',
}

# 1. 合法语法全量通过（6 种切点 × 12 语法 ≈ 144 断言）
assert_valid_call(
    "tagged", VALID["tagged"], "shell", {"command": "ls -la /tmp", "intent": "查临时目录"})
assert_valid_call(
    "bare", VALID["bare"], "file_write",
    {"content": 'line1\nline2 "quoted"', "path": "/tmp/a.py"})
assert_valid_call("bare_flat", VALID["bare_flat"], "web_search",
                  {"query": "claude 检测", "max_results": 7})
assert_valid_call("loose_kv", VALID["loose_kv"], "shell",
                  {"command": "ps aux | head -5", "intent": "查进程"})
assert_valid_call("loose_kv_inline", VALID["loose_kv_inline"], "file_read",
                  {"path": "/etc/hosts"})
# fn_expr_multi：连续两个表达式全部回收（兜底四或闸门打捞）
for chunk in (1, 3, 7, 17, 64, None):
    content, calls = run_parse(VALID["fn_expr"], chunk)
    got = [json.loads(c["function"]["arguments"]) for c in calls]
    check(f"fn_expr_multi[chunk={chunk}]",
          len(calls) == 2 and all(c["function"]["name"] == "web_search" for c in calls)
          and got[0].get("max_results") == 7 and got[1].get("query") == "glm 退化",
          f"n={len(calls)}")
    check(f"fn_expr_multi[chunk={chunk}]无残骸", not leak_markers(content),
          f"content={content[:80]!r}")
assert_valid_call("colon", VALID["colon"], "web_search", {"query": "claude fingerprint 检测方法"})
assert_valid_call("attr", VALID["attr"], "shell",
                  {"command": "ls /tmp 2>/dev/null", "intent": "查"})
assert_valid_call("tagged_escape", VALID["tagged_escape"], "shell",
                  {"command": 'echo "---" && ls'})
assert_valid_call("tagged_unicode", VALID["tagged_unicode"], "shell",
                  {"command": "echo 中文测试", "note": "中文"})
assert_valid_call("loose_kv_crlf", VALID["loose_kv_crlf"], "shell", {"command": "uptime"})

# tagged_multi：两个调用都解析出
for chunk in (1, 3, 7, 17, 64, None):
    content, calls = run_parse(VALID["tagged_multi"], chunk)
    check(f"tagged_multi[chunk={chunk}]",
          len(calls) == 2 and calls[0]["function"]["name"] == "shell"
          and calls[1]["function"]["name"] == "file_read", f"n={len(calls)}")

# ---------------------------------------------------------------- 变异 fuzz

MUT_STEPS = 24  # 每条语料的截断采样点数（控制总量）


def mutations(genome):
    """对一条合法语料生成变异集合（确定性）。"""
    muts = []
    body = genome["text"]
    # M1 逐点截断（含首尾附近加密采样）
    cuts = sorted({int(i * (len(body) - 1) / MUT_STEPS) for i in range(MUT_STEPS + 1)} | {1, 5, len(body) - 1})
    for c in cuts:
        if 0 < c < len(body):
            muts.append(("trunc", body[:c]))
    # M2 删一个结构字符
    for ch in "{}\"\\:<>,":
        idxs = [i for i, x in enumerate(body) if x == ch]
        if idxs:
            i = _rng.choice(idxs)
            muts.append((f"drop{ch!r}", body[:i] + body[i + 1:]))
    # M3 去 \$ 转义类：把合法 \" 换成 "（未转义双引号）
    if '\\"' in body:
        i = body.index('\\"')
        muts.append(("unesc", body[:i] + '"' + body[i + 2:]))
    # M4 非法转义注入（markdown \$ / 正则 \d）
    i = _rng.randrange(len(body))
    muts.append(("inj\\$", body[:i] + "\\$" + body[i:]))
    muts.append(("inj\\d", body[:i] + "\\d" + body[i:]))
    # M5 协议噪音注入
    for tok in ("<path>", "</arg_value>", "<tool_callback>"):
        j = _rng.randrange(len(body))
        muts.append((f"inj{tok}", body[:j] + tok + body[j:]))
    # M6 行重组：join / 首行前插空行
    muts.append(("join", body.replace("\n", " ")))
    muts.append(("blankline", body.replace("\n", "\n\n", 1)))
    return muts


FUZZ_GENOMES = {
    "tagged": VALID["tagged"],
    "bare": VALID["bare"],
    "bare_flat": VALID["bare_flat"],
    "loose_kv": VALID["loose_kv"],
    "loose_kv_inline": VALID["loose_kv_inline"],
    "fn_expr": VALID["fn_expr"],
    "colon": VALID["colon"],
    "attr": VALID["attr"],
    "tagged_escape": VALID["tagged_escape"],
}

# 2. 变异矩阵：非流式 + 两种流式切点（≈ 9 语料 × ~35 变异 × 3 路径）
for gname, gtext in FUZZ_GENOMES.items():
    for mname, mutated in mutations({"text": gtext}):
        assert_no_leak(f"fuzz:{gname}:{mname}", mutated)
        assert_no_leak(f"fuzz:{gname}:{mname}:s3", mutated, chunk=3)
        assert_no_leak(f"fuzz:{gname}:{mname}:s17", mutated, chunk=17)

# ---------------------------------------------------------------- 定向矩阵

# 3. 截断修复专项：裸 JSON 逐状态截断（每个切点都不得泄漏）
TRUNC_BASE = ('前文。\n{"name": "file_write", "arguments": {"content": "a\\nb \\"q\\"",'
              ' "path": "/tmp/x.py", "timeout": 60}}\n后文不会有了')
for c in range(TRUNC_BASE.index('{"name"'), len(TRUNC_BASE), 9):
    assert_no_leak(f"trunc_scan:{c}", TRUNC_BASE[:c])
assert_no_leak("trunc:仅开括号", '前文。\n{"')
assert_no_leak("trunc:名字一半", '前文。\n{"name": "file_wr')
assert_no_leak("trunc:键一半", '前文。\n{"name": "file_write", "argu')
assert_no_leak("trunc:值一半", '{"name": "shell", "arguments": {"command": "ls -l')
assert_no_leak("trunc:转义截断", '{"name": "shell", "arguments": {"command": "echo a\\')
assert_no_leak("trunc:unicode转义截断", '{"name": "shell", "arguments": {"note": "\\u4e')

# 截断后打捞出的参数必须保留（值完整性抽检）
rest, calls = _parse_tool_calls(
    '{"name": "file_write", "arguments": {"content": "a\\nb", "path": "/tmp/x.py"}', KNOWN)
check("trunc打捞:参数完整", len(calls) == 1 and json.loads(
    calls[0]["function"]["arguments"]) == {"content": "a\nb", "path": "/tmp/x.py"},
    f"calls={calls}")

# 4. 非法转义专项（JSON 解码后须无损还原原文；合法 \ 序列走标准解码）
for esc, want in (
    (r"\$", r"\$"), (r"\d", r"\d"), (r"\w", r"\w"), (r"\.", r"\."),
    (r"\-", r"\-"), (r"\ ", r"\ "), ("\\", "\\"),  # 孤立反斜杠
    (r"\u12", r"\u12"), (r"\uZZZZ", r"\uZZZZ"),
):
    body = ('{"name": "shell", "arguments": {"command": "echo ' + esc + ' data",'
            ' "intent": "x"}}')
    content, calls = _parse_tool_calls(body, KNOWN)
    ok = len(calls) == 1 and json.loads(
        calls[0]["function"]["arguments"])["command"] == f"echo {want} data"
    check(f"esc:{esc!r}", ok and not leak_markers(content),
          f"n={len(calls)} content={content[:60]!r}")
    # 流式极限切点（chunk=1：每个转义对/引号都跨 chunk 分裂）下参数仍须
    # 完整——增量扫描器的转义对在 chunk 边界断裂时不得错算引号/深度
    for chunk in (1, 2, 5):
        sp = _StreamToolCallSplitter(KNOWN)
        out = []
        for k in range(0, len(body), chunk):
            out.append(sp.feed(body[k:k + chunk]))
        tail, scalls = sp.flush()
        got = json.loads(scalls[0]["function"]["arguments"])["command"] if scalls else None
        check(f"esc:{esc!r}:s{chunk}", len(scalls) == 1 and got == f"echo {want} data"
              and not leak_markers("".join(out) + tail),
              f"n={len(scalls)} got={got!r}")

# 5. 未转义双引号专项（deepseek 风格 echo "---"）
for q in ('echo "---"', 'echo "{" && ls', 'grep "x" f && echo "}"'):
    body = '{"name": "shell", "arguments": {"command": "' + q + '", "intent": "查"}}'
    assert_no_leak(f"quote:{q}", body)
    content, calls = _parse_tool_calls(body, KNOWN)
    if calls:
        got = json.loads(calls[0]["function"]["arguments"]).get("command", "")
        check(f"quote:{q}:修复保真", q in got, f"got={got!r}")

# 6. 散装 kv 专项变体
LOOSE_VARIANTS = [
    ('name: "shell"\narguments: {"command": "ls"}', "shell", {"command": "ls"}),
    ('name: "shell" arguments: {"command": "ls"}', "shell", {"command": "ls"}),
    ('name: "shell",\narguments: {"command": "ls"}', "shell", {"command": "ls"}),
    ('  name: "shell"\n  arguments: {"command": "ls"}', "shell", {"command": "ls"}),
    ('name:\t"shell"\narguments:\t{"command": "ls"}', "shell", {"command": "ls"}),
    ('name: "shell"\narguments: {\n  "command": "ls"\n}', "shell", {"command": "ls"}),
    ('name: "file_write"\narguments: {"content": "a\\"b", "path": "/t"}', "file_write", {"content": 'a"b', "path": "/t"}),
]
for i, (body, want_name, want_args) in enumerate(LOOSE_VARIANTS):
    assert_valid_call(f"loose_var{i}", body, want_name, want_args, chunks=(1, 5, 23, None))

# 散装 kv 截断：不得泄漏
LOOSE_FULL = 'name: "shell"\narguments: {"command": "ps aux | grep ethan", "intent": "查"}'
for c in range(LOOSE_FULL.index("arguments"), len(LOOSE_FULL), 7):
    assert_no_leak(f"loose_trunc:{c}", LOOSE_FULL[:c])

# 7. 散装 kv 误伤防护：未知工具 / 非工具语境
for prose in (
    'name: "myapp"\n  port: 8080\n  host: localhost',
    'name: "未知名"\narguments: {"x": 1}',
    'name: 张三的配置说明',
    'The name: "field" is required.',
    'name: "shell"\n但是下一行根本不是 arguments，只是普通说明文字。',
):
    assert_prose_kept(f"loose_fp:{prose[:20]!r}", prose)

# 8. 闸门专项：修不动的已知工具调用必须被丢弃（不泄漏），未知名保留
gate_broken = '说明。\n{"name": "shell", "arguments" "command": "ls"} 完。'
content, calls = _parse_tool_calls(gate_broken, KNOWN)
check("gate:已知坏调用丢弃", '"arguments"' not in content and "说明。" in content and "完。" in content,
      f"content={content!r}")

gate_unknown = '配置示例：\n{"name": "server", "arguments": {"port": 8080}}\n完。'
content, calls = _parse_tool_calls(gate_unknown, KNOWN)
check("gate:未知名正文保留", "server" in content and len(calls) == 0, f"content={content!r}")

# 9. 正文不误伤语料（不变量 B 全量）
PROSE = [
    '# 配置说明\n```yaml\nname: "web"\nreplicas: 3\n```',
    '{"name": " deepseek-chat", "arguments": {"model": "v3"}, "id": "x"}',
    'SVG 例子：<svg><path d="M0 0 L10 10"/></svg> 保留。',
    'Python 里 `print("hello")` 就够了。',
    'shell 语法：`ls -la | grep tmp`。',
    'JSON Schema 的 "name" 字段是必填的，"arguments" 是可选的。',
    'web_search 是一个工具: query 参数必填。',
    'CRLF 正文\r\n第二行 name: "普通"\r\n第三行',
    '带 <path> 标签的 XML 文档：<config><path>/tmp</path></config>',
    'const config = { name: "app", args: () => {} }',
    '他说："name": "shell" 只是文档里的示例字段。',
]
for p in PROSE:
    assert_prose_kept(f"prose:{p[:16]!r}", p)

# 10. 混合流：好调用 + 后续坏调用（闸门补收/丢弃，不整段泄漏）
mixed = ('先说明。\n' + O + '\n{"name": "shell", "arguments": {"command": "pwd"}}\n' + C
         + '\n{"name": "file_read", "arguments": {"path": "/etc/host')
for chunk in (None, 5, 31):
    content, calls = run_parse(mixed, chunk)
    check(f"mixed[chunk={chunk}]", len(calls) == 2
          and calls[1]["function"]["name"] == "file_read"
          and not leak_markers(content),
          f"n={len(calls)} marks={leak_markers(content)}")

# 11. 空参 / 嵌套 / 数组参数
for body, want in (
    ('{"name": "shell", "arguments": {}}', {}),
    ('{"name": "shell", "arguments": {"opts": {"a": [1, 2, {"b": 3}]}}}', {"opts": {"a": [1, 2, {"b": 3}]}}),
    ('{"name": "shell", "arguments": {"list": ["x", "y"]}}', {"list": ["x", "y"]}),
):
    content, calls = _parse_tool_calls(body, KNOWN)
    ok = len(calls) == 1 and json.loads(calls[0]["function"]["arguments"]) == want
    check(f"args_shape:{want}", ok, f"calls={calls}")

# 12. 稳定性：超大 payload + 深嵌套不炸、不超时（隐性性能回归）
big = ('{"name": "file_write", "arguments": {"content": "'
       + ("x" * 200_000) + '", "path": "/tmp/big"}}')
assert_no_leak("big_payload", big, chunk=1024)

# 12b. 大 payload 流式路径耗时上限（隐性性能回归曾达二次方放大：
# 200KB@512 曾 15.7s，增量扫描+尾窗后应 <1s；此处给 5 倍余量防 CI 抖动）
import time as _time
_big512 = ('{"name": "file_write", "arguments": {"content": "'
           + ("x" * 200_000) + '", "path": "/tmp/big"}}')
_sp = _StreamToolCallSplitter(KNOWN)
_t0 = _time.perf_counter()
for _k in range(0, len(_big512), 512):
    _sp.feed(_big512[_k:_k + 512])
_sp.flush()
_big_dt = _time.perf_counter() - _t0
check("big_payload:512chunk 耗时<2.5s", _big_dt < 2.5, f"took={_big_dt:.2f}s")

print()
print(f"total checks: {total}")
print("RESULT:", "ALL PASS" if not fails else f"FAILED({len(fails)}): {fails[:8]}")
sys.exit(1 if fails else 0)
