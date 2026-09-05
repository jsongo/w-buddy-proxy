"""文本协议工具调用解析器：标签/属性/裸 JSON/散装 KV/冒号连写全形态解析。

上游不支持（历史版本头下的实测误判，现作为 glm-5-turbo 等非原生通道模型的
兜底）结构化 tool_calls 时，模型按提示词教学把调用语法输出在正文里；本模块
把这些文本形态还原成 OpenAI tool_calls。含 JSON 修复、流式跨 chunk 分片
（_StreamToolCallSplitter）与泄漏闸门。
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

log = logging.getLogger(__name__)

# 工具调用流式标签（教学格式 + Trae 原生格式的标签名）
# tool_callback：deepseek-v4-flash 实测的伪回调头（开标签 + 意图文本、不闭合），
# 不扣留的话会当正文流出（下游 UI 直接看到 <tool_callback>…）；扣留后交给
# _parse_tool_calls 统一剥离
_TOOL_STREAM_TAGS = ("tool_call", "tool_calls", "tool_action", "tool_name", "tool_callback", "command", "arg_key", "arg_value", "seed:tool_call", "function", "parameter")

# 裸 JSON 工具调用候选前缀：模型偶尔省略标签，直接在行首输出
# {"name": ...} 或 [{"name": ...}]（数组包多个调用）
# deepseek-v4-pro 偶尔在 { 和 " 之间加空格：{ "name": ... }
_BARE_CANDS = ('{"name"', '{ "name"', '[{"name"', '[{ "name"', '[ {"name"')
_BARE_RE = re.compile(r'(?m)^[ \t]*(\[\s*\{\s*"name"|\{\s*"name")')

# XML 属性风格工具调用：<tool_call name="..." command="..." />
# 属性值里可能含 '>'（如 shell 命令 2>/dev/null），不能用简单正则匹配整标签，
# 用起始正则定位 + 引号感知扫描
_ATTR_TAG_START_RE = re.compile(r"<(?:tool_call|tool_calls|tool_action)\b", re.I)
_ATTR_VAL_RE = re.compile(r"(\w+)\s*=\s*(?:\"([^\"]*)\"|'([^']*)')")

# 函数调用表达式参数值：k="v" / k='v' / k=7（数字）/ k=true|false|null（裸字面量）
_KV_VAL_RE = re.compile(
    r"(\w+)\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|(-?\d+(?:\.\d+)?)|(true|false|null))",
    re.I,
)

# 行首函数调用表达式（裸函数语法兜底的流式扣留锚点）
_LINE_CALL_EXPR_RE = re.compile(r"(?m)^[ \t]*([A-Za-z_][\w.]*)[ \t]*\(")

_CALL_NAME_RE = re.compile(r"([A-Za-z_][\w.]*)\s*\(")


def _parse_kv_args(body: str) -> dict[str, Any]:
    """解析函数调用表达式参数体 k=v, k2=v2 -> dict（保留键原大小写）。

    相比 _ATTR_VAL_RE 额外支持未加引号的数字/布尔/null（glm-5.3 实测输出
    web_search(query="...", max_results=7, ...)，max_results 不带引号，
    旧正则会直接丢掉这个参数）。
    """
    args: dict[str, Any] = {}
    for m in _KV_VAL_RE.finditer(body):
        key = m.group(1)
        if m.group(2) is not None:
            val: Any = m.group(2)
        elif m.group(3) is not None:
            val = m.group(3)
        elif m.group(4) is not None:
            f = float(m.group(4))
            val = int(f) if f.is_integer() else f
        else:
            val = {"true": True, "false": False, "null": None}[
                m.group(5).lower()]
        args[key] = val
    return args


def _find_call_exprs(
    text: str,
    names: frozenset[str] | set[str] | None = None,
) -> list[tuple[str, dict[str, Any], int, int]]:
    """扫描文本里的函数调用表达式 name(k=v, ...)。

    引号感知 + 括号配对扫描（参数值里可能含括号/逗号），连续多个调用
    （空格/换行分隔）逐个返回。names 传入时只接受已知工具名（裸文本
    防误伤）；None 时任意名字都收（调用块内模型已显式标记是调用）。
    返回 [(name, args_dict, start, end), ...]，end 为右括号后一位。
    """
    results: list[tuple[str, dict[str, Any], int, int]] = []
    pos = 0
    n = len(text)
    while True:
        m = _CALL_NAME_RE.search(text, pos)
        if not m:
            break
        name = m.group(1)
        # 引号感知扫描到配对的右括号
        j = m.end()
        depth = 1
        quote: str | None = None
        end = -1
        while j < n:
            ch = text[j]
            if quote is not None:
                if ch == quote:
                    quote = None
            elif ch in ('"', "'"):
                quote = ch
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    end = j
                    break
            j += 1
        if end == -1:
            # 括号不闭合（截断流）：跳过名字继续扫
            pos = m.end()
            continue
        if names is None or name in names:
            body = text[m.end():end]
            args = _parse_kv_args(body)
            if not args:
                args = {"input": body.strip()}
            results.append((name, args, m.start(), end + 1))
        pos = end + 1
    return results


def _colon_marker_re(tools: frozenset[str] | set[str], anchored: bool = True) -> re.Pattern:
    """glm-5.3 连写变体「工具名+参数名+冒号」的起始标记正则。

    模型省略括号/引号/等号，输出 web_searchquery: <自由文本>；工具名与
    参数名之间无空格。按工具名长度降序拼接分支避免前缀撞名。
    anchored=True 时首个标记必须在行首；连续调用时下一个标记会与
    上一个值粘连（methodsweb_searchquery:），用 anchored=False 续扫。
    命名捕获组：tool=工具名，param=参数名。
    """
    alt = "|".join(
        sorted((re.escape(t) for t in tools), key=len, reverse=True))
    body = rf"(?P<tool>(?:{alt}))(?P<param>[a-z_][a-z0-9_]*)\s*:"
    if anchored:
        return re.compile(rf"(?m)^[ \t]*{body}")
    return re.compile(body)


def _find_colon_joined_calls(
    text: str,
    tools: frozenset[str] | set[str],
) -> list[tuple[str, dict[str, Any], int, int]]:
    """解析 web_searchquery: <自由文本> 连写形态的工具调用。

    glm-5.3 实测：模型连括号都省掉，直接写
    ``web_searchquery: how to ... methodsweb_searchquery: 检测套壳...``
    （连续多个时上一个值直接拼到下一个标记前，无分隔符）。
    首个标记必须位于行首；后续标记由上一个值的结束边界界定。
    返回 [(name, args_dict, start, end), ...]，end 为值结束位置。
    """
    first = _colon_marker_re(tools).search(text)
    if not first:
        return []
    marks = [first]
    follow_re = _colon_marker_re(tools, anchored=False)
    while True:
        m2 = follow_re.search(text, marks[-1].end())
        if not m2:
            break
        marks.append(m2)
    results: list[tuple[str, dict[str, Any], int, int]] = []
    n = len(text)
    for i, mk in enumerate(marks):
        val_start = mk.end()
        val_end = marks[i + 1].start() if i + 1 < len(marks) else n
        value = text[val_start:val_end]
        # 去掉值末尾的 markdown 水平分隔线行（如模型另起一行写的 ---）
        value = re.sub(r"(?m)^[ \t]*-{3,}[ \t]*$", "", value)
        # 自由文本压成单行
        value = re.sub(r"\s+", " ", value).strip(" \t\r\n-")
        if not value:
            continue
        results.append((
            mk.group("tool"),
            {mk.group("param"): value},
            mk.start("tool"),
            val_end,
        ))
    return results


def _extract_attr_calls(rest: str, calls: list[dict[str, Any]]) -> str:
    """提取 XML 属性风格的工具调用标签，返回剩余文本。

    形如 <tool_call name="shell" command="ls" intent="..." />（自闭合或
    开标签均可）。属性值内的 '>' 不会截断标签（扫描时跳过引号内字符）。
    """
    out: list[str] = []
    pos = 0
    n = len(rest)
    while True:
        m = _ATTR_TAG_START_RE.search(rest, pos)
        if not m:
            out.append(rest[pos:])
            break
        i = m.start()
        # 扫描到真正的 '>'（跳过引号内的字符）
        j = i + 1
        quote = None
        end = -1
        while j < n:
            ch = rest[j]
            if quote is not None:
                if ch == quote:
                    quote = None
            elif ch in ('"', "'"):
                quote = ch
            elif ch == ">":
                end = j
                break
            j += 1
        if end == -1:
            out.append(rest[pos:])
            break
        tag_text = rest[i:end + 1]
        attrs: dict[str, str] = {}
        for k, v1, v2 in _ATTR_VAL_RE.findall(tag_text):
            attrs[k.lower()] = v1 if v2 == "" else v2
        name = attrs.get("name") or attrs.get("tool") or ""
        if not name:
            # 没有名字（教学格式的开标签/复数容器壳）：保留原文交给后续清理
            out.append(rest[pos:end + 1])
            pos = end + 1
            continue
        raw = attrs.get("command") or attrs.get("arguments") or attrs.get("args") or ""
        if raw.strip().startswith("{"):
            try:
                args = json.loads(raw)
            except ValueError:
                args = {"command": raw}
        else:
            args = {"command": raw}
            if attrs.get("intent"):
                args["intent"] = attrs["intent"]
        out.append(rest[pos:i])
        calls.append(_mk_tool_call(
            len(calls), name, json.dumps(args, ensure_ascii=False)))
        pos = end + 1
    return "".join(out)


def _tool_names(tools: list[dict[str, Any]] | None) -> frozenset[str] | None:
    """从 OpenAI tools 定义里提取工具名集合。"""
    if not tools:
        return None
    names = set()
    for t in tools:
        f = t.get("function") if isinstance(t, dict) else None
        if isinstance(f, dict) and f.get("name"):
            names.add(f["name"])
    return frozenset(names) or None


def _valid_bare_call_obj(obj: Any, known_tools: frozenset[str]) -> bool:
    """裸 JSON 是否是合法的工具调用对象。

    严格校验（避免误伤正文里的普通 JSON）：name 必须是已知工具名。
    两种形态放行：
    - 标准形态：带 arguments/args，且除 name/tool/intent 外无杂键
      （intent 是 ethan 教的说明字段，放行）；
    - 扁平形态（glm-5.3 实测）：name 与参数平铺——
      {"name": "web_search", "query": "...", "max_results": 10}，
      无 arguments 包裹层，但至少带一个参数键。
    """
    if not isinstance(obj, dict):
        return False
    name = obj.get("name")
    if not isinstance(name, str) or name not in known_tools:
        return False
    if "arguments" in obj or "args" in obj:
        return set(obj) <= {"name", "arguments", "args", "intent", "tool"}
    # 扁平形态：name/tool 之外还有键即视为参数平铺
    return bool(set(obj) - {"name", "tool"})


def _flatten_call_args(obj: dict[str, Any]) -> dict[str, Any]:
    """从裸 JSON 调用对象提取参数；扁平形态（无 arguments 层）取剩余键。

    glm-5.3 还会把 arguments 输出成 JSON 字符串（而非对象）：
    {"name": "web_search", "arguments": "{\"query\": \"...\", \"max_results\": 10}"}
    —— 解包成 dict，避免下游拿到 {"input": "<整段 JSON 字符串>"}。
    """
    args = obj.get("arguments", obj.get("args"))
    if args is None:
        args = {k: v for k, v in obj.items() if k not in ("name", "tool")}
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            if isinstance(parsed, dict):
                args = parsed
        except Exception:
            pass
    return args if isinstance(args, dict) else {"input": args}


def _quote_is_close(s: str, i: int) -> bool:
    """Heuristic: does the ``"`` at position *i* close the current string?

    Tolerates unescaped quotes inside values (deepseek v4 pro leak): a quote
    followed by ``}`` / ``]`` only closes when valid structure tokens follow
    the bracket — a ``}`` immediately followed by more string content
    (e.g. ``echo "}" && ls``) is an inner quote, not the end of the value.
    """
    n = len(s)
    j = i + 1
    while j < n and s[j] in " \t":
        j += 1
    if j >= n:
        return True
    c = s[j]
    if c in ":,\n":
        return True
    if c in "}]":
        k = j + 1
        while k < n and s[k] in " \t\r\n":
            k += 1
        if k >= n or s[k] in ",}]\n":
            return True
        return False
    return False


def _find_obj_extent(text: str, start: int) -> int:
    """Best-effort brace-depth scan to find the end of a JSON object.

    Returns the index *after* the matching ``}`` or -1 if unbalanced.
    Tracks string state with :func:`_quote_is_close` so stray braces inside
    broken string values (e.g. ``"command": "echo "}" && ls"``) don't
    prematurely zero the depth.  If the boundary still can't be found the
    caller degrades gracefully (no repair, raw text preserved).
    """
    if start >= len(text) or text[start] != "{":
        return -1
    depth = 0
    in_string = False
    i = start
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\\" and i + 1 < n:
            i += 2
            continue
        if ch == '"':
            if not in_string:
                in_string = True
            elif _quote_is_close(text, i):
                in_string = False
            i += 1
            continue
        if not in_string:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i + 1
        i += 1
    return -1


def _repair_json_quotes(s: str) -> str | None:
    """Try to fix unescaped double-quotes inside JSON string values.

    DeepSeek v4 pro occasionally emits shell commands with unescaped ``"``
    (e.g. ``echo "---"``), which makes the JSON invalid.  Walk through the
    text tracking string-state; quotes that don't look like a real string
    close (per :func:`_quote_is_close`) are escaped in place.

    Returns the repaired string, or *None* if no repair was applied or the
    input doesn't look like a JSON object/array.
    """
    stripped = s.lstrip()
    if not stripped or stripped[0] not in "{[":
        return None
    out: list[str] = []
    i = 0
    n = len(s)
    in_string = False
    repaired_any = False
    while i < n:
        ch = s[i]
        if not in_string:
            out.append(ch)
            if ch == '"':
                in_string = True
            i += 1
            continue
        if ch == "\\" and i + 1 < n:
            out.append(s[i : i + 2])
            i += 2
            continue
        if ch == '"':
            if _quote_is_close(s, i):
                out.append(ch)
                in_string = False
            else:
                out.append('\\"')
                repaired_any = True
            i += 1
            continue
        out.append(ch)
        i += 1
    if not repaired_any:
        return None
    return "".join(out)


_JSON_VALID_ESCAPES = set('"\\/bfnrtu')
_JSON_HEX = set("0123456789abcdefABCDEF")


def _repair_invalid_escapes(s: str) -> str:
    r"""修复 JSON 字符串值里非法的 \ 转义：\X -> \\X。

    glm-5.3-flash 实测（s_20260804_0200_eb1d）：command 值里嵌 python /
    正则脚本，脚本自身的 \d \s . 等单反斜杠序列对 JSON 是非法转义
    （json.loads 报 Invalid \escape），现有引号修复管不了这类。把非法
    \X 双写成 \\X，解码后无损还原为原文 \X。
    """
    if "\\" not in s:
        return s
    out: list[str] = []
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if ch == "\\":
            if i + 1 >= n:
                # 尾部孤立反斜杠（截断流常见）：按字面量处理
                out.append("\\\\")
                i += 1
                continue
            nxt = s[i + 1]
            if nxt == "u":
                hexrun = s[i + 2:i + 6]
                if len(hexrun) == 4 and all(c in _JSON_HEX for c in hexrun):
                    out.append(s[i:i + 6])
                    i += 6
                    continue
            elif nxt in _JSON_VALID_ESCAPES:
                out.append(s[i:i + 2])
                i += 2
                continue
            out.append("\\\\")
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _repair_truncated_json(s: str) -> Any | None:
    """修复被截断的 JSON 对象/数组（流断在结构中间）。

    实测（s_20260905_0141_3a19）：流断在裸调用尾部，最外层 } 缺失。
    策略：字符串状态机扫描记录括号栈；结束仍在字符串里 → 补闭合引号；
    按栈深逆序补 } / ] 后尝试解析。仍失败则逐步回退到最后几个结构
    分隔符（, : { } [ ]，字符串外的）之前重试——会丢尾部半截键值，
    但保住前面已完整的参数。返回解析结果，修不动返回 None。
    """
    stripped = s.strip()
    if not stripped or stripped[0] not in "{[":
        return None

    def scan(text: str) -> tuple[list[str], bool, list[int]]:
        stack: list[str] = []
        seps: list[int] = []
        in_string = False
        i, n = 0, len(text)
        while i < n:
            ch = text[i]
            if ch == "\\" and i + 1 < n:
                i += 2
                continue
            if ch == '"':
                if not in_string:
                    in_string = True
                elif _quote_is_close(text, i):
                    in_string = False
                i += 1
                continue
            if not in_string:
                if ch in "{[":
                    stack.append("}" if ch == "{" else "]")
                    seps.append(i)
                elif ch in "}]":
                    if stack:
                        stack.pop()
                    seps.append(i)
                elif ch in ",:":
                    seps.append(i)
            i += 1
        return stack, in_string, seps

    fixed = _repair_invalid_escapes(stripped)
    _, in_string, seps = scan(fixed)
    base = fixed + '"' if in_string else fixed
    # 从破坏最小的候选开始：整段补齐括号 → 逐个回退到结构分隔符之前
    cut_points = [len(base)]
    cut_points += [p for p in reversed(seps) if p < len(base)][:6]
    for cut in dict.fromkeys(cut_points):
        frag = base[:cut]
        st, ins, _ = scan(frag)
        if ins:
            frag += '"'
            st, _, _ = scan(frag)
        try:
            return json.loads(frag + "".join(reversed(st)), strict=False)
        except ValueError:
            continue
    return None


def _repair_call_json(s: str) -> tuple[Any, int] | None:
    """统一修复管线：坏 JSON 调用 → (解析结果, 消费到的原始偏移)。

    覆盖三类实测损坏（此前都会整段泄漏进正文）：
    a) 非法 \\ 转义（脚本/正则的单反斜杠）——_repair_invalid_escapes；
    b) 字符串值里未转义双引号——_repair_json_quotes；
    c) 尾部截断（缺最外层 } / 断在字符串中间）——_repair_truncated_json。
    边界平衡（_find_obj_extent 找得到）走 a+b 后 strict=False 解析
    （容忍字面换行）；不平衡走 c。返回 None 表示修不动。
    """
    if not s or s[0] not in "{[":
        return None
    raw_end = _find_obj_extent(s, 0)
    if raw_end != -1:
        fixed = _repair_invalid_escapes(s[:raw_end])
        qr = _repair_json_quotes(fixed)
        if qr is not None:
            fixed = qr
        try:
            return json.loads(fixed, strict=False), raw_end
        except ValueError:
            pass
        return None
    obj = _repair_truncated_json(s)
    if obj is None:
        return None
    return obj, len(s)


def _peek_call_name(s: str) -> str:
    """从坏调用原文里提取工具名（修复全失败时判断是否丢弃的依据）。"""
    m = re.search(r'"(?:name|tool)"\s*:\s*"([^"\n]*)"', s)
    return m.group(1) if m else ""


def _loose_kv_plausible(seg: str, tools: frozenset[str]) -> bool:
    """seg（从行首 name 起）是否仍是散装调用形态的前缀（含完整形态）。

    形态：?name?[ \t]*: "工具名" <空白/逗号/换行> arguments[ \t]*: {...
    name 键可带引号（实测变体：{"name": "x", "arguments": {...}} 外层
    大括号被删后剩下行首 "name": ...）。逐段消费，任一段与允许的前缀
    不符即发散。工具名已闭合但不在已知列表 → 发散（放行，正文里的
    YAML/示例不吞）。
    """
    seg = seg.lstrip(" \t")
    if not seg:
        return False
    if seg[0] == '"':
        # 引号键：必须是 "name" 的发展前缀（"na / "name / "name" / "name":）
        kw = '"name"'
        if not kw.startswith(seg):
            if not seg.startswith(kw):
                return False
            seg = seg[len(kw):]
        else:
            return True
    else:
        if not seg.startswith("name"):
            return bool(seg) and "name".startswith(seg)
        seg = seg[4:]
        if seg.startswith('"'):
            seg = seg[1:]        # name": 形态（外层 { 被删的损坏变体）
    n = len(seg)

    def eat_ws(j: int) -> int:
        while j < n and seg[j] in " \t":
            j += 1
        return j

    i = eat_ws(0)
    if i >= n:
        return True          # 冒号还在路上
    if seg[i] != ":":
        return False
    i = eat_ws(i + 1)
    if i >= n:
        return True
    if seg[i] != '"':
        return False
    q = seg.find('"', i + 1)
    if q == -1:
        return "\n" not in seg[i:]   # 工具名未闭合：跨行即发散
    name = seg[i + 1:q]
    if not name or name not in tools:
        return False
    # 工具名之后：空白/逗号/换行 + arguments 关键字（带不带引号都认）
    k = q + 1
    while k < n and seg[k] in " \t\r\n,":
        k += 1
    if k >= n:
        return True          # 等待 arguments
    if seg[k] == '"':
        k += 1
        if k >= n:
            return True
    if not seg.startswith("arguments", k):
        return "arguments".startswith(seg[k:])
    k += len("arguments")
    if k < n and seg[k] == '"':
        k += 1
    k = eat_ws(k)
    if k >= n:
        return True
    if seg[k] != ":":
        return False
    k = eat_ws(k + 1)
    if k >= n:
        return True
    return seg[k] == "{"     # 对象已开：扣到 flush 统一解析


class _StreamToolCallSplitter:
    """流式工具调用切分器：正文按 chunk 透传，工具调用块整段扣留。

    调用块（无论是否已闭合）都要在 flush 时统一解析成 OpenAI tool_calls，
    不能当正文放行——所以从首个疑似标签起全部扣留（与泄漏清洗器的
    "闭合即放行"语义不同）。

    known_tools：请求携带的工具名集合。传入后额外扣留「行首裸 JSON」候选
    （模型偶尔省略标签、直接输出 {"name": ...} 裸调用，实测 deepseek-v4-pro
    连续多轮调用后会出现这种偷懒写法）。
    """

    def __init__(self, known_tools: frozenset[str] | None = None) -> None:
        self._buf = ""
        self._tools = frozenset(known_tools) if known_tools else None
        self._bare_hold = None

    @staticmethod
    def _first_tag_pos(buf: str) -> int:
        low = buf.lower()
        best = -1
        for tag in _TOOL_STREAM_TAGS:
            start = 0
            # 开标签 + 闭标签都要扣（</arg_value> 类闭合残骸实测会夹正文流出）
            for opener in ("<" + tag, "</" + tag):
                while True:
                    i = low.find(opener, start)
                    if i == -1:
                        break
                    after = low[i + len(opener): i + len(opener) + 1]
                    if after in ("", ">", "/", " ", "\t", "\n"):
                        if best == -1 or i < best:
                            best = i
                        break
                    start = i + 1
        # 部分标签前缀（跨 chunk 分裂 / 标签名中间被塞了残骸标签）：取最早
        # 的未闭合 "<"。片段取到下一个 "<" 或 ">" 为止——用 rfind 取最后
        # 一个 "<" 会把前面的半截标签（如 "<tool_c<tool_callback>all>"
        # 里的 "<tool_c"）当安全文本放出去
        j = low.find("<")
        while j != -1:
            nxt_lt = low.find("<", j + 1)
            frag = buf[j + 1: nxt_lt if nxt_lt != -1 else len(buf)]
            if ">" in frag:
                if nxt_lt == -1:
                    break
                j = nxt_lt
                continue
            partial = frag.lower().lstrip("/ \t")
            if any(t.startswith(partial) for t in _TOOL_STREAM_TAGS):
                if best == -1 or j < best:
                    best = j
            break
        return best

    def _bare_pos(self, buf: str) -> int:
        """行首裸 JSON 调用候选位置（含跨 chunk 分裂的前缀）。"""
        if not self._tools:
            return -1
        best = -1
        m = _BARE_RE.search(buf)
        if m:
            best = m.start(1)
        # 尾部行可能是分裂中的候选前缀（如 buf 以 '{"na' 结尾）；整候选开头
        # （如 '{"name"' 恰好等于尾行）也扣——flush 端 _BARE_RE 能处理
        last_nl = buf.rfind("\n")
        line = buf[last_nl + 1:]
        ls = line.lstrip()
        if ls and any(c.startswith(ls) or ls.startswith(c) for c in _BARE_CANDS):
            p = last_nl + 1 + (len(line) - len(ls))
            if best == -1 or p < best:
                best = p
        return best

    def _loose_kv_pos(self, buf: str) -> int:
        """行首散装 name:/arguments: 调用的扣留位置（deepseek-v4-flash 实测）。

        形态：name: "shell"
              arguments: {"command": ...}
        外层大括号与标签全省；name 键可带引号（外层 { 被删的损坏变体）。
        行首 name 且后续与该形态前缀吻合时扣留；发散（name 非已知工具、
        后续不是 arguments:、工具名跨行未闭合）立即返回 -1 放行，
        误伤窗口只有一两行。
        """
        if not self._tools:
            return -1
        for m in re.finditer(r"(?m)^[ \t]*\"?name\b", buf):
            # 限窗切片：plausible 只消费形态前缀，无需 buf 全尾（无界切片
            # 在多 name 行的大 payload 下是二次方放大，实测 69KB/2000 行 870ms）
            if _loose_kv_plausible(buf[m.start():m.start() + 2048], self._tools):
                return m.start()
        # 尾部行可能是发展中的 name 前缀（跨 chunk 分裂，如 buf 以 'na' 结尾）
        last_nl = buf.rfind("\n")
        line = buf[last_nl + 1:]
        ls = line.lstrip()
        if ls and len(ls) <= len('"name"') and (
                '"name"'.startswith(ls) or "name".startswith(ls)):
            return last_nl + 1 + (len(line) - len(ls))
        return -1

    def _inline_intent_pos(self, buf: str) -> int:
        """行内强证据调用的扣留位置（正文与调用连写无换行的变体）。

        与闸门的行内模式一致：已知工具名 + arguments 键 / 参数赋值同现，
        单独出现工具名不扣（正文里提到工具名很常见）。
        """
        if not self._tools:
            return -1
        if not hasattr(self, "_inline_re"):
            # tools_alt 与 _tail_hold_pos 共用惰性缓存（避免每 chunk 重拼正则串）
            tools_alt = getattr(self, "_tools_alt", None)
            if not tools_alt:
                tools_alt = "|".join(
                    sorted((re.escape(t) for t in self._tools), key=len, reverse=True))
                self._tools_alt = tools_alt
            self._inline_re = re.compile(
                r'\{\s*"?name"?\s*:[ \t]*"(?:' + tools_alt + r')"[^\n]{0,200}?"arguments"'
                r'|name\s*:\s*"(?:' + tools_alt + r')"[ \t\r\n]*,?[ \t\r\n]*arguments\s*:'
                r'|\b(?:' + tools_alt + r')[ \t]*\([ \t]{0,40}[A-Za-z_]\w*[ \t]*=')
        m = self._inline_re.search(buf)
        return m.start() if m else -1

    def _call_expr_pos(self, buf: str) -> int:
        """行首已知工具名调用表达式的扣留位置（glm-5.3 裸函数语法变体）。

        模型偶尔连 <tool_call> 标签都省掉，直接在行首输出
        web_search(query="...", ...)（可连续多个，尾随孤立 </arg_value>）。
        行首 + 已知工具名 + 紧跟 "(" 的组合在正文里极其罕见，值得扣留。
        含跨 chunk 分裂的前缀（尾行是纯工具名前缀、可能 ( 在下一个 chunk）。
        """
        if not self._tools:
            return -1
        best = -1
        for m in _LINE_CALL_EXPR_RE.finditer(buf):
            if m.group(1) in self._tools:
                if best == -1 or m.start() < best:
                    best = m.start()
        # 尾部行可能是分裂中的候选前缀（如 buf 以行首 'web_sear' 结尾）
        last_nl = buf.rfind("\n")
        line = buf[last_nl + 1:]
        ls = line.lstrip()
        if ls and re.fullmatch(r"[A-Za-z_][\w.]*", ls) and len(ls) <= max(
                len(t) for t in self._tools):
            if any(t.startswith(ls) for t in self._tools):
                p = last_nl + 1 + (len(line) - len(ls))
                if best == -1 or p < best:
                    best = p
        return best

    def _colon_pos(self, buf: str) -> int:
        """行首冒号连写调用的扣留位置（glm-5.3 web_searchquery: 变体）。

        完整标记 web_searchquery: 直接正则定位；跨 chunk 分裂的前缀
        （web_searchqu / web_searchquery / web_searchquery:）用尾行
        前缀匹配扣住。
        """
        if not self._tools:
            return -1
        if not hasattr(self, "_colon_re"):
            self._colon_re = _colon_marker_re(self._tools)
        best = -1
        m = self._colon_re.search(buf)
        if m:
            best = m.start("tool")
        last_nl = buf.rfind("\n")
        line = buf[last_nl + 1:]
        ls = line.lstrip()
        if ls:
            for t in self._tools:
                if ls.startswith(t):
                    tail = ls[len(t):]
                    if tail and re.fullmatch(r"[a-z0-9_]*:?", tail):
                        p = last_nl + 1 + (len(line) - len(ls))
                        if best == -1 or p < best:
                            best = p
                        break
        return best

    # 「发展中前缀」三正则天然只关心尾部（合法匹配最长 ~250 字符），限定
    # 窗口防止每 chunk 对全量 buffer 回溯——实测不限定时二次方放大
    # （500KB 扣留 payload 流式路径 105s CPU，200KB 15.7s）。
    _TAIL_WINDOW = 512

    def _tail_hold_pos(self, buf: str) -> int:
        """尾部发展中候选的扣留位置（行内调用证据天然跨 chunk）。

        正文与调用连写（无换行）时，「{"name": "tool"...arguments」这类
        证据要若干 chunk 才凑齐——按完整证据扣留会先把开头当正文放出。
        这里检查缓冲区尾部是否仍是某个调用形态的"发展前缀"：是则从候选
        起点扣住，下一个 chunk 发散（变成普通正文）立即释放；到 flush 还
        扣着就交给统一解析/闸门。尾部有界（几十~几百字符），误扣不会
        无限拖延输出；到 flush 仍未发散的尾部由解析/闸门统一处置。
        """
        if not self._tools:
            return -1
        if not hasattr(self, "_tail_res"):
            tools_alt = "|".join(
                sorted((re.escape(t) for t in self._tools), key=len, reverse=True))
            # arguments 的逐字符可选发展：a(?:r(?:g(?:...(?:s)?)?)...)?
            arg_dev = "a" + "".join("(?:%s" % ch for ch in "rguments") + ")?" * 8
            self._tools_alt = tools_alt
            self._tail_res = (
                # {"name 逐字符发展（值可有可无——{ 一出现就扣，发散即放）
                re.compile(r'\{\s*"?\s*(?:n(?:a(?:m(?:e)?)?)?)?\s*"?\s*'
                           r':?[ \t]*"?(?:[A-Za-z_][\w.\-]{0,40})?"?$'),
                # name: "tool" [\\s,]* arguments 发展（值可有可无——
                # name: 一出现就扣，发散即放）
                re.compile(r'\bname"?\s*:[ \t]*"?(?:[A-Za-z_][\w.\-]{0,40})?"?'
                           r'[ \t\r\n,]*(?:' + arg_dev + r')?\s*:?\s*\{?$'),
                # 已知工具名 + ( 参数发展
                re.compile(r'\b(?:' + tools_alt + r')\s*\(\s*[^)\n]{0,200}$'),
            )
            self._bare_anchor_re = re.compile(
                r'\{\s*"?name"?\s*:\s*"?(?:' + tools_alt + r')')
        best = -1
        # 尾部正则只在窗口内搜（$ 锚定，窗口外命中不可能也不需要）
        offset = max(0, len(buf) - self._TAIL_WINDOW)
        tail = buf[offset:]
        for pat in self._tail_res:
            m = pat.search(tail)
            if m:
                p = offset + m.start()
                if best == -1 or p < best:
                    best = p
        # 裸 JSON 参数区深度扣留（不限长，大 payload 的 content 可达数百 KB）：
        # {"name": "已知工具" 前缀一旦出现，扣住直到参数区闭合或流结束。
        # 增量扫描版：锚点只搜一次，之后每 chunk 只扫新增尾部
        bp = self._bare_hold_pos(buf)
        if bp != -1 and (best == -1 or bp < best):
            best = bp
        # 工具名前缀尾部（含单字符、完整名）：tool( 表达式跨 chunk 发展；
        # 完整名后跟非 ( 时下一步即发散，误扣窗口一两个 chunk
        m2 = re.search(r'[A-Za-z_][\w.]{0,31}$', tail)
        if m2 and any(t.startswith(m2.group(0)) for t in self._tools):
            p = offset + m2.start()
            if best == -1 or p < best:
                best = p
        return best

    def _bare_hold_pos(self, buf: str) -> int:
        """裸 JSON 扣留的增量扫描实现（替代每 chunk 全量 anchor+extent）。

        状态（_bare_hold）以 buf 绝对坐标保存：扣留期间 feed 只会从 pos≤at
        处切片（at 是本检测器的扣留点，feed 取各检测器最小值），锚点之前
        不会被放出，因此坐标只需在 feed 切片时整体平移（_shift_bare_hold）。
        闭合释放后进入 closed 态：锚点只在新增尾部（scan 之后）重搜，避免
        对已闭合区域反复全量扫描。
        """
        st = self._bare_hold
        if st is not None and st.get("closed"):
            m = self._bare_anchor_re.search(buf, st["scan"])
            if not m:
                st["scan"] = len(buf)
                return -1
            st = self._bare_hold = {
                "at": m.start(), "b": -1, "scan": m.end(),
                "depth": 0, "instr": False, "closed": False,
            }
        elif st is None:
            m = self._bare_anchor_re.search(buf)
            if not m:
                return -1
            st = self._bare_hold = {
                "at": m.start(), "b": -1, "scan": m.end(),
                "depth": 0, "instr": False, "closed": False,
            }
        at = st["at"]
        i, n = st["scan"], len(buf)
        depth, instr, b = st["depth"], st["instr"], st["b"]
        while i < n:
            ch = buf[i]
            if b == -1:
                # arguments 对象的 '{' 还没出现
                if ch == "{":
                    b, depth = i, 1
                i += 1
                continue
            if instr:
                if ch == "\\":
                    if i + 1 >= n:
                        # 反斜杠是本 buffer 最后一字符：转义对跨 chunk 分裂，
                        # scan 原地停在反斜杠处，下 chunk 重读后再跳过整对
                        break
                    i += 2
                    continue
                if ch == '"':
                    instr = False
            elif ch == '"':
                instr = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth <= 0:
                    # 对象闭合：释放扣留（闭合的完整对象交给正常解析路径）
                    st.update(scan=i + 1, depth=depth, instr=instr, b=b, closed=True)
                    return -1
            i += 1
        st.update(scan=i, depth=depth, instr=instr, b=b)
        return at

    def _shift_bare_hold(self, pos: int, reset: bool) -> None:
        """feed() 切片后平移裸扣留状态的 buf 坐标；buffer 重置时清状态。"""
        if reset or pos <= 0 or not self._bare_hold:
            if reset:
                self._bare_hold = None
            return
        st = self._bare_hold
        st["at"] -= pos
        st["scan"] -= pos
        if st["b"] != -1:
            st["b"] -= pos
        if st["at"] < 0:  # 防御：正常不会发生（pos ≤ at）
            self._bare_hold = None

    def feed(self, text: str) -> str:
        self._buf += text
        pos = self._first_tag_pos(self._buf)
        if self._tools:
            bp = self._bare_pos(self._buf)
            if bp != -1 and (pos == -1 or bp < pos):
                pos = bp
            ep = self._call_expr_pos(self._buf)
            if ep != -1 and (pos == -1 or ep < pos):
                pos = ep
            cp = self._colon_pos(self._buf)
            if cp != -1 and (pos == -1 or cp < pos):
                pos = cp
            lp = self._loose_kv_pos(self._buf)
            if lp != -1 and (pos == -1 or lp < pos):
                pos = lp
            tp = self._tail_hold_pos(self._buf)
            if tp != -1 and (pos == -1 or tp < pos):
                pos = tp
            ip = self._inline_intent_pos(self._buf)
            if ip != -1 and (pos == -1 or ip < pos):
                pos = ip
        if pos == -1:
            safe, self._buf = self._buf, ""
            self._shift_bare_hold(0, reset=True)
        else:
            safe, self._buf = self._buf[:pos], self._buf[pos:]
            self._shift_bare_hold(pos, reset=False)
        return safe

    def flush(self) -> tuple[str, list[dict[str, Any]]]:
        rest, calls = _parse_tool_calls(self._buf, self._tools)
        self._buf = ""
        self._bare_hold = None
        return rest, calls

# prompt-based function calling 的教学格式标签。选 XML 标签 + JSON 参数：
# 标签与模型原生 SOLO 协议同构（遵循度最高），JSON 参数适配任意工具 schema
_TC_OPEN = '<tool_call>'
_TC_CLOSE = '</tool_call>'
_TOOL_DECODER = json.JSONDecoder()

# 散装键值调用（deepseek-v4-flash 实测 s_20260902_2130_c8bd）：外层大括号
# 与标签全省，name: "shell" 与 arguments: {...} 直接散装两行
_LOOSE_KV_RE = re.compile(
    r'(?m)^[ \t]*name[ \t]*:[ \t]*"(?P<name>[^"\n]+)"'
    r'[ \t\r\n]*,?[ \t\r\n]*arguments[ \t]*:[ \t]*\{')

# 泄漏闸门的证据提取：从调用开头段里取工具名。宽松匹配——坏调用的
# name 键本身可能已损坏（缺引号/缺冒号/值截断），行首调用语法开头 +
# 已知工具名的组合已经足够强，键的形态不苛求
_GATE_NAME_RES = (
    re.compile(r'[ \t]*(?:\{[\s]*|[{,][ \t]*)"?name"?[ \t]*:[ \t]*"?([A-Za-z_][\w.-]*)'),
    re.compile(r'[ \t]*name"?[ \t]*:[ \t]*"?([A-Za-z_][\w.-]*)'),
)

# word_tool 兜底的搜索窗口：工具名作为「调用开头」证据，只在段首找
_GATE_WORD_WINDOW = 120


def _gate_tool_name(seg: str) -> str:
    """闸门证据提取：从调用开头段里取出工具名（JSON 键 / 散装 name:）。"""
    for pat in _GATE_NAME_RES:
        m = pat.match(seg)
        if m:
            return m.group(1)
    return ""


def _gate_word_tool(seg: str, known: frozenset[str]) -> str:
    """证据兜底：段里出现独立成词的已知工具名（键语法损坏时）。

    只在段首 _GATE_WORD_WINDOW 内找：工具名属于「调用开头」的证据，
    离 `{`/锚点太远的命中（比如参数值或后文提到别的工具名）不足以
    把整段判成调用——误伤正文会静默发伪 tool_call，比泄漏更糟。
    """
    head = seg[:_GATE_WORD_WINDOW]
    for t in known:
        if re.search(r"(?<![A-Za-z0-9_])" + re.escape(t) + r"(?![A-Za-z0-9_])", head):
            return t
    return ""


def _gate_expr_end(rest: str, p: int) -> int:
    """函数表达式残段的结束位置：引号感知扫到匹配的 ')'，扫不到取行尾。"""
    i = rest.find("(", p)
    if i == -1:
        eol = rest.find("\n", p)
        return eol if eol != -1 else len(rest)
    depth = 0
    quote = None
    j = i
    n = len(rest)
    while j < n:
        ch = rest[j]
        if quote is not None:
            if ch == "\\":
                j += 2
                continue
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return j + 1
        elif ch == "\n" and depth == 0:
            return j
        j += 1
    return n


def _gate_loose_block_end(rest: str, p: int) -> int:
    """无对象的散装残段结束位置：name 行 + 后随的 arguments 前缀行。

    只在两类情况下允许丢弃：a) name 行之后跟着 arguments（含其截断前缀）
    开头的行——参数区已开始；b) name 行是全文最后一行——流断在调用开头。
    其他情况（name 行后面是普通正文）返回 -1 不动，避免吞正文。
    """
    eol = rest.find("\n", p)
    if eol == -1:
        return len(rest)          # b) 截断到全文尾
    end = eol
    k = eol + 1
    while k < len(rest):
        eol2 = rest.find("\n", k)
        line = rest[k:eol2 if eol2 != -1 else len(rest)]
        ls = line.lstrip(" \t")
        if not ls:
            break
        if "arguments:".startswith(ls) or ls.startswith("arguments"):
            end = eol2 if eol2 != -1 else len(rest)
            k = end + 1
            # arguments: 行带出内容（如无大括号的散参数）就到此为止
            if not ls.startswith("arguments") or len(ls) > len("arguments"):
                break
            continue
        break
    if end == eol:
        # name 行后面不是 arguments：仅当其余部分是纯空白（name 行是最后
        # 一个非空行，流断在调用开头）才丢；有正文则不动
        if not rest[eol + 1:].strip():
            return len(rest)
        return -1
    return end


def _mk_tool_call(idx: int, name: str, arguments: str) -> dict[str, Any]:
    return {
        "id": f"call_{idx:02d}_{uuid.uuid4().hex[:20]}",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _parse_tool_calls(
    content: str,
    known_tools: frozenset[str] | set[str] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """从模型输出里解析工具调用块，返回 (剩余正文, tool_calls 列表)。

    主路径解析教学格式（JSON 参数，用 raw_decode 正确处理嵌套大括号）；
    兜底一：Trae 原生 SOLO XML（<tool_name>/<command>，名字对不上由
    下游 agent 报错纠偏）；
    兜底二：无标签裸 JSON（known_tools 提供时启用——模型连续多轮调用后
    会偷懒省略标签，直接在行首输出 {"name": ...}）。
    """
    # 容错归一化：模型偶尔把标签写成复数变体（开/闭合都可能，大小写不定），
    # 实测出现过「标准单数开标签 + 复数闭标签」的混搭——严格匹配会解析失败，
    # 兜底清理只剥开标签、留下复数闭标签和裸 JSON 整段泄漏进正文。
    # 先统一归一成教学的标准单数标签再进主解析。
    content = re.sub(r"</\s*tool_calls\s*>", _TC_CLOSE, content, flags=re.I)
    content = re.sub(r"<\s*tool_calls\s*>", _TC_OPEN, content, flags=re.I)
    # 伪回调头 <tool_callback>（deepseek-v4-flash 实测）必须在主解析前剥离：
    # 闭合块整块剥；未闭合剥到下一个 <tool_call> 为止——主循环会把
    # <tool_call> 消费掉，放后面做前瞻就没有锚点了。剥法与 _LEAK_RE 统一。
    content = re.sub(r"<tool_callback[^>]*>.*?</tool_callback>", "", content, flags=re.S)
    content = re.sub(r"<tool_callback[^>]*>.*?(?=<tool_call[>\s])", "", content, flags=re.S)
    # seed 协议块（doubao-seed-evolving 实测 s_20260905_0233_f291）：
    # <seed:tool_call>\n<function name="shell"><parameter name="command"
    # string="true">gh api ...</parameter><parameter name="intent"
    # string="true">...</parameter></function>...\n</seed:tool_call>
    # 参数值是标签体（命令含引号/反引号都不转义），按 XML 提取而非 JSON。
    # 未闭合（流截断）剥到全文尾。先于主循环做：调用量计入 calls 后，
    # 其余兜底的 not calls 门槛自然跳过
    calls: list[dict[str, Any]] = []

    def _grab_seed_block(m: re.Match) -> str:
        got_seed = False
        for fm in re.finditer(
                r'<function\s+name="([^"]+)"[^>]*>([\s\S]*?)</function>',
                m.group(1), re.I):
            args = {pm.group(1): pm.group(2) for pm in re.finditer(
                r'<parameter\s+name="([^"]+)"[^>]*>([\s\S]*?)</parameter>',
                fm.group(2), re.I)}
            calls.append(_mk_tool_call(
                len(calls), fm.group(1), json.dumps(args, ensure_ascii=False)))
            got_seed = True
        if got_seed:
            return ""
        # 块里没有完整 function（截断半截）：丢弃壳、留参数文本给闸门判断
        return m.group(1)

    content = re.sub(
        r"<seed:tool_call>\s*([\s\S]*?)(?:</seed:tool_call>|\Z)",
        _grab_seed_block, content, flags=re.I)
    # 记录原文是否带 arg_key/arg_value 残骸：glm-5.3 的坏调用块伴随孤立
    # </arg_value>，这是「这段文本确实是坏掉的调用」而非正文代码示例的强信号
    had_arg_debris = bool(re.search(r"</?\s*arg_(?:key|value)", content))
    rest_parts: list[str] = []
    pos = 0
    n = len(content)
    while pos < n:
        i = content.find(_TC_OPEN, pos)
        if i == -1:
            rest_parts.append(content[pos:])
            break
        rest_parts.append(content[pos:i])
        j = i + len(_TC_OPEN)
        block_calls: list[tuple[str, str]] = []
        closed = False
        expr_consumed = False  # 块内表达式路径已整块消费
        # 块内循环：连续解码多个 JSON（复数容器装多个调用），直到闭标签
        while True:
            while j < n and content[j] in " \t\r\n":
                j += 1
            if j >= n:
                break
            if content.startswith(_TC_CLOSE, j):
                j += len(_TC_CLOSE)
                closed = True
                break
            if content[j] == "{":
                try:
                    obj, end = _TOOL_DECODER.raw_decode(content, j)
                except ValueError:
                    # 修复段限制在块内（</tool_call> 之前）；块未闭合（流
                    # 截断）时扩到全文尾，交给截断修复
                    close = content.find(_TC_CLOSE, j)
                    seg = content[j:close] if close != -1 else content[j:]
                    rep = _repair_call_json(seg)
                    if rep is not None:
                        obj, rend = rep
                        if isinstance(obj, dict):
                            nm = obj.get("name") or obj.get("tool") or ""
                            ag = obj.get("arguments", obj.get("args", {}))
                            if not isinstance(ag, dict):
                                ag = {"input": ag}
                            if nm:
                                block_calls.append(
                                    (str(nm), json.dumps(ag, ensure_ascii=False)))
                                j = j + rend
                                continue
                    break
                name = obj.get("name") or obj.get("tool") or ""
                args = obj.get("arguments", obj.get("args", {}))
                if not isinstance(args, dict):
                    args = {"input": args}
                if name:
                    block_calls.append(
                        (str(name), json.dumps(args, ensure_ascii=False)))
                j = end
                continue
            # 不是 JSON：可能是 glm-5.3 的函数调用表达式写法——
            # <tool_call>web_search(query="...", max_results=7) web_search(...)</arg_value></tool_call>
            # （块内可连续多个调用表达式，混孤立 </arg_value> 残骸）。
            # 引号感知扫到块尾，全部转成 tool_calls 后整块消费。
            close = content.find(_TC_CLOSE, j)
            block_text = content[j:close if close != -1 else n]
            exprs = _find_call_exprs(block_text, None)
            if exprs:
                for name, args, _s, _e in exprs:
                    calls.append(_mk_tool_call(
                        len(calls), name, json.dumps(args, ensure_ascii=False)))
                pos = (close + len(_TC_CLOSE)) if close != -1 else n
                expr_consumed = True
                break
            break  # 既不是 JSON 也不是闭标签：块不合法
        # closed=正常闭合；j>=n=块内耗尽全文（流截断在块中间）——
        # 两种情况都要提交已修复出的调用，截断的未闭合块不再整块泄漏
        if (closed or j >= n) and block_calls:
            for name, args_json in block_calls:
                calls.append(_mk_tool_call(len(calls), name, args_json))
            pos = j
            continue
        if expr_consumed:
            continue
        # 不是合法调用块：保留原文继续扫描
        rest_parts.append(content[i:i + len(_TC_OPEN)])
        pos = i + len(_TC_OPEN)
    rest = "".join(rest_parts)

    # 残余噪音剥离（无论是否已解析出调用都要做）：
    # - 孤立 tool_callback 标签壳：未闭合到流尾的（模型放弃调用继续说正文）
    #   只剥壳、保留内部文本，不能整段吞掉
    # - 孤立 arg_key / arg_value 标签：glm-5.3 函数调用语法残骸
    rest = re.sub(r"</?\s*tool_callback[^>]*>", "", rest)
    rest = re.sub(r"</?\s*arg_(?:key|value)[^>]*>", "", rest)
    rest = re.sub(r"\n{3,}", "\n\n", rest)

    # 兜底：Trae 原生 <tool_name>/<command>（仅当教学格式没解析出任何调用时）
    if not calls:
        # 兜底三：函数调用语法（glm-5.3 实测：无视 JSON 教学，标签内直接写
        # skill_read(intent="...")，常混入孤立 </arg_value> 残骸）
        def _grab_fn_call(match: re.Match) -> str:
            name, body = match.group(1), match.group(2)
            args = _parse_kv_args(body)
            if not args:
                args = {"input": body.strip()}
            calls.append(_mk_tool_call(
                len(calls), name.strip(), json.dumps(args, ensure_ascii=False)))
            return ""

        # ) 与 </tool_call> 之间允许夹孤立残骸标签（</arg_value> 等）
        fn_call_re = re.compile(
            r"<tool_call>\s*([A-Za-z_][\w.]*)\s*\(([\s\S]*?)\)\s*"
            r"(?:</[^>]+>\s*)*</tool_call>",
            re.I,
        )
        rest = fn_call_re.sub(_grab_fn_call, rest)
        if calls:
            return rest.strip(), calls

        def _grab_native(match: re.Match) -> str:
            calls.append(_mk_tool_call(len(calls), match.group(1), json.dumps(
                {"command": match.group(2).strip()}, ensure_ascii=False)))
            return ""

        native_re = re.compile(
            r"<tool_name>\s*([^<\s]+)\s*</tool_name>\s*<command>\s*([\s\S]*?)\s*</command>",
            re.S,
        )
        rest = native_re.sub(_grab_native, rest)

        # 兜底：XML 属性风格 <tool_call name="..." command="..." />
        # （实测 deepseek-v4-pro 偶发输出这种自闭合属性标签）
        rest = _extract_attr_calls(rest, calls)
        rest = re.sub(r"</?tool_action[^>]*>", "", rest)
        rest = re.sub(re.escape(_TC_OPEN) + r"\s*", "", rest)
        rest = re.sub(r"\s*" + re.escape(_TC_CLOSE), "", rest)

    # 兜底二：无标签裸 JSON（仅 agent 请求、且前两种格式都没解析出调用）
    if not calls and known_tools:
        known = frozenset(known_tools)
        # 同一行连续多个裸 JSON（deepseek-v4-pro 实测）：在 } {"name" 边界
        # 插入换行，使后续对象也能被 _BARE_RE 的行首锚点匹配到
        rest = re.sub(r'}\s+(?=\{\s*"name")', '}\n', rest)
        rest_parts2: list[str] = []
        pos = 0
        while True:
            m = _BARE_RE.search(rest, pos)
            if not m:
                rest_parts2.append(rest[pos:])
                break
            js = m.start(1)
            try:
                obj, end = _TOOL_DECODER.raw_decode(rest, js)
            except ValueError:
                rep = _repair_call_json(rest[js:])
                if rep is not None:
                    obj, rend = rep
                    items = obj if isinstance(obj, list) else [obj]
                    if all(_valid_bare_call_obj(it, known) for it in items):
                        rest_parts2.append(rest[pos:m.start()].rstrip())
                        for it in items:
                            nm = it.get("name") or it.get("tool") or ""
                            ag = _flatten_call_args(it)
                            calls.append(_mk_tool_call(
                                len(calls), str(nm),
                                json.dumps(ag, ensure_ascii=False)))
                        pos = js + rend
                        continue
                # 修复失败：name 是已知工具 → 丢弃坏原文（实测坏调用整段
                # 泄进正文纯噪音，agent 侧重试即可）；未知 name 可能是正文
                # 里的普通 JSON，保留原文
                peek = _peek_call_name(rest[js:])
                if peek and peek in known:
                    rest_parts2.append(rest[pos:m.start()].rstrip())
                    end_drop = _find_obj_extent(rest, js)
                    if end_drop == -1:
                        # 边界不明：有界丢弃——最后一个 } 后的尾巴像正文
                        # （短、无 JSON 语法字符）则保留，否则吞到文尾防泄漏
                        seg2 = rest[js:js + 4096]
                        lb = seg2.rfind("}")
                        tail2 = rest[js + lb + 1:] if lb != -1 else ""
                        if lb != -1 and len(tail2) <= 60 and not re.search(
                                r'[{}\[\]":,]|\\', tail2):
                            end_drop = js + lb + 1
                        else:
                            end_drop = len(rest)
                    pos = end_drop
                    continue
                rest_parts2.append(rest[pos:m.end()])
                pos = m.end()
                continue
            items = obj if isinstance(obj, list) else [obj]
            if obj and all(_valid_bare_call_obj(it, known) for it in items):
                rest_parts2.append(rest[pos:m.start()].rstrip())
                for it in items:
                    name = it.get("name") or it.get("tool") or ""
                    args = _flatten_call_args(it)
                    calls.append(_mk_tool_call(
                        len(calls), str(name), json.dumps(args, ensure_ascii=False)))
                pos = end
            else:
                rest_parts2.append(rest[pos:m.end()])
                pos = m.end()
        rest = "".join(rest_parts2)

    # 兜底 2.5：散装键值形态（deepseek-v4-flash 实测 s_20260902_2130_c8bd）：
    # name: "shell" 与 arguments: {...} 散装两行。行首 name + 已知工具名 +
    # arguments: { 的组合在正文里极罕见，误伤风险低
    if not calls and known_tools:
        known = frozenset(known_tools)
        loose_out: list[str] = []
        lpos = 0
        for lm in _LOOSE_KV_RE.finditer(rest):
            if lm.group("name") not in known:
                continue
            j = lm.end() - 1  # 指向 arguments 对象的 '{'
            try:
                args_obj, lend = _TOOL_DECODER.raw_decode(rest, j)
            except ValueError:
                rep = _repair_call_json(rest[j:])
                if rep is None:
                    continue
                args_obj, lend = rep[0], min(j + rep[1], len(rest))
            if not isinstance(args_obj, dict):
                continue
            loose_out.append(rest[lpos:lm.start()])
            calls.append(_mk_tool_call(
                len(calls), lm.group("name"),
                json.dumps(args_obj, ensure_ascii=False)))
            lpos = lend
        if lpos:
            rest = "".join(loose_out) + rest[lpos:]

    # 兜底四：裸函数调用表达式（glm-5.3 实测 Scopus 会话变体：
    # web_search(query="...", max_results=7) web_search(...)</arg_value>
    # ——无 <tool_call> 包裹、连续多个调用 + 孤立 </arg_value> 残骸）。
    # 门槛（防误伤正文里的普通代码示例）：
    # a) 前面所有格式都没解出调用；b) 工具名在请求的 known_tools 里；
    # c) 原文带 arg_* 残骸（坏调用块的强信号），或表达式位于行首
    # （流式 splitter 对行首已知工具名调用会主动扣留，两端约定一致）。
    if not calls and known_tools and (had_arg_debris or _LINE_CALL_EXPR_RE.search(rest)):
        known = frozenset(known_tools)
        exprs = _find_call_exprs(rest, known)
        # 有残骸信号时全部收；没有残骸时只收行首的（inline 的可能是正文示例）
        if not had_arg_debris:
            exprs = [e for e in exprs
                     if e[2] == 0 or rest[e[2] - 1] == "\n"]
        if exprs:
            out: list[str] = []
            pos3 = 0
            for name, args, s, e in exprs:
                out.append(rest[pos3:s])
                calls.append(_mk_tool_call(
                    len(calls), name, json.dumps(args, ensure_ascii=False)))
                pos3 = e
            out.append(rest[pos3:])
            rest = "".join(out)

    # 兜底五：冒号连写形态（glm-5.3 实测：web_searchquery: <自由文本>，
    # 模型省略括号/引号/等号；连续多个时上一个值直接拼到下一个标记前）。
    # 门槛与裸表达式一致：仅在已知工具名、且前面格式都没解出调用时启用；
    # 首个标记必须在行首（splitter 端会主动扣留这类行首前缀）。
    if not calls and known_tools:
        colon_calls = _find_colon_joined_calls(rest, frozenset(known_tools))
        if colon_calls:
            out: list[str] = []
            pos5 = 0
            for name, args, s, e in colon_calls:
                out.append(rest[pos5:s])
                calls.append(_mk_tool_call(
                    len(calls), name, json.dumps(args, ensure_ascii=False)))
                pos5 = e
            out.append(rest[pos5:])
            rest = "".join(out)
            rest = re.sub(r"(?m)^[ \t]*-{3,}[ \t]*\n?", "", rest)

    # glm-5.3 实测：正文带 <think>...</think>\n\n</think>（多一个游离闭标签）。
    # agent 模式不走 _sanitize_agent_leak，think 清洗在这里兜住。
    if "<think>" in rest or "</think>" in rest:
        rest = re.sub(r"<think>.*?</think>", "", rest, flags=re.S)
        rest = re.sub(r"</?\s*think>", "", rest)
        rest = rest.strip()

    # 泄漏闸门（最后一道防线）：所有解析与修复都跑完后，残文里仍有
    # 「调用语法证据 + 已知工具名」的段——说明出现了未知的退化变体或
    # 修复全失败。先给合法对象最后一次补收机会（兜底二/2.5 在已有
    # calls 时不再跑，混合流里后续裸调用会漏到这里）；补收不了则丢弃
    # 该段（坏调用原文对下游是纯噪音，agent 下一轮重试即可），保住前后
    # 正文，并打 WARN——新变体从 proxy.log 搜 [toolcall-gate] 即可自动
    # 发现，不用等用户在聊天里撞见。
    if known_tools:
        known = frozenset(known_tools)
        # 先剥掉孤立 tool_call 标签壳（含损坏形态：开标签断 > 、只剩
        # 半截的），否则「JSON 已被兜底收走、壳留在正文」照样是泄漏。
        # 放在所有兜底之后，不影响 <tool_call> 块的正常解析
        rest = re.sub(r"</?\s*tool_call\b[^>\n]{0,40}>?", "", rest)
        tools_alt = "|".join(sorted((re.escape(t) for t in known), key=len, reverse=True))
        # 候选开头：行首（裸 JSON / 散装 name: / 引号键散装 / 标签族 /
        # 函数表达式）+ 行内强证据（正文与调用连写无换行：必须工具名与
        # arguments 键/参数赋值同现，防止误伤正文里恰好提到工具名的句子）
        gate_re = re.compile(
            r'(?m)^[ \t]*(?:'
            r'\{[\s]*"?name"?[ \t]*:[ \t]*"'
            r'|\{(?=[^\n]{0,80}"arguments")'
            r'|\{[\s]*"name"(?=[^\n]{0,100}(?:' + tools_alt + r'))'
            r'|name"?[ \t]*:[ \t]*"'
            r'|"(?:name|tool)"?[ \t]*:[ \t]*"'
            r'|</?\s*t(?:ool_action|ool_name|ool_callback|ool_call)[>\s]'
            r'|(?P<fnl>[A-Za-z_][\w.]*)[ \t]*\('
            r')'
            r'|\{\s*"?name"?\s*:[ \t]*"(?:' + tools_alt + r')"[^\n]{0,200}?"arguments"'
            r'|name"?\s*:[ \t]*"(?:' + tools_alt + r')"[ \t\r\n]*,?[ \t\r\n]*arguments\s*:'
            r'|\b(?P<fni>(?:' + tools_alt + r'))[ \t]*\([ \t]{0,40}[A-Za-z_]\w*[ \t]*=')
        gate_out: list[str] = []
        gpos = 0
        touched = False
        for gm in gate_re.finditer(rest):
            p = gm.start()
            if p < gpos:
                continue
            seg = rest[p:p + 512]
            fnm = gm.group("fnl") or gm.group("fni")
            if fnm:
                nm = fnm
                if nm not in known:
                    continue
                # 表达式：引号感知扫到行内匹配的 ')'，扫不到取行尾
                end = _gate_expr_end(rest, p)
                # 打捞机会：完整键值参数表达式直接收成调用
                exprs = _find_call_exprs(rest[p:end], known)
                if exprs:
                    gate_out.append(rest[gpos:p])
                    for name, args, _s2, _e2 in exprs:
                        calls.append(_mk_tool_call(
                            len(calls), name, json.dumps(args, ensure_ascii=False)))
                    gpos = end
                    touched = True
                    continue
            else:
                nm = _gate_tool_name(seg) or _gate_word_tool(seg, known)
                if not nm or nm not in known:
                    continue
                b = seg.find("{")
                if b == -1:
                    # 无对象：散装截断（name 行后面 arguments 还没来/坏了）。
                    # 只在「后随 arguments 前缀行」或「name 行已是全文最后一
                    # 行」（流截断）时才丢，避免吞正文里提到工具名的 YAML
                    end = _gate_loose_block_end(rest, p)
                    if end == -1:
                        continue
                else:
                    b_abs = p + b
                    e = _find_obj_extent(rest, b_abs)
                    end = e if e != -1 else len(rest)
                    # 补收机会：对象完整且能解码 → 直接收成调用。
                    # 对象本身带 name 键 = 裸 JSON 本体；不带 = 散装 kv 的
                    # arguments 对象（外层 { 被删的实测变体）。
                    # extent 找不到（结构引号双损）或解码失败 → 修复管线
                    # 再试一次（截断尾流实测能打捞出完整调用）
                    obj = None
                    if e != -1:
                        try:
                            cand, _ = _TOOL_DECODER.raw_decode(rest, b_abs)
                            obj = cand if isinstance(cand, dict) else None
                        except ValueError:
                            rep = _repair_call_json(rest[b_abs:e])
                            if rep is not None and isinstance(rep[0], dict):
                                obj = rep[0]
                    else:
                        # 边界不明（结构/引号双损）：有界丢弃——到段内最后
                        # 一个 } 为止，且其后的尾巴要像正文（短、无 JSON
                        # 语法）才保留，否则吞到文尾防泄漏
                        last_b = seg.rfind("}")
                        tail = rest[p + last_b + 1:] if last_b != -1 else ""
                        if last_b != -1 and len(tail) <= 60 and not re.search(
                                r'[{}\[\]":,]|\\', tail):
                            end = p + last_b + 1
                        else:
                            end = len(rest)
                        rep = _repair_call_json(rest[b_abs:end])
                        if rep is not None and isinstance(rep[0], dict) and (
                                rep[0].get("name") or rep[0].get("tool")
                                or any(k2 not in ("name", "tool") for k2 in rep[0])):
                            obj = rep[0]
                    if obj is not None:
                        call_nm = str(obj.get("name") or obj.get("tool") or nm)
                        if call_nm in known:
                            # 证据已足够强（调用开头 + 已知工具名 + 可解析对象），
                            # 放宽杂键限制——收回比丢弃/泄漏都好
                            ag = _flatten_call_args(obj) if obj.get("name") or obj.get("tool") else obj
                            calls.append(_mk_tool_call(
                                len(calls), call_nm, json.dumps(ag, ensure_ascii=False)))
                            # 打捞与丢弃同样入日志：误收伪调用比泄漏更难排查，
                            # [toolcall-gate] 一个 grep 应看到闸门的全部动作
                            log.warning(
                                "[toolcall-gate] 补收调用残段 tool=%s len=%d 片段=%r",
                                call_nm, end - p, rest[p:p + 80])
                        gate_out.append(rest[gpos:p])
                        gpos = end
                        touched = True
                        continue
                    if e == -1:
                        # 修不动且边界不明：有界丢弃——到段内最后一个 } 或
                        # " 为止，绝不吞后面的正文
                        last_b = max(seg.rfind("}"), seg.rfind('"'))
                        end = p + last_b + 1 if last_b != -1 else min(p + 256, len(rest))
            gate_out.append(rest[gpos:p])
            gpos = end
            touched = True
            log.warning(
                "[toolcall-gate] 丢弃无法解析的调用残段 tool=%s len=%d 片段=%r",
                nm, end - p, rest[p:p + 80])
        if touched:
            gate_out.append(rest[gpos:])
            rest = "".join(gate_out)
            # 丢弃后留下的孤立标签壳 / 参数残骸一并清掉
            rest = re.sub(r"</?\s*(?:tool_call|tool_callback|tool_action"
                          r"|tool_name|arg_key|arg_value)[^>]*>", "", rest)

    return rest.strip(), calls

