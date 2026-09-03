"""回归验证：ethan 会话 s_20260903_1613_155c 实测 badcase。

用例全部取自 2026-09-03 codebuddy-proxy.jsonl 的真实 upstream 输出。
"""
import sys

sys.path.insert(0, "src")
from codebuddy_proxy.trae_provider import (  # noqa: E402
    _parse_tool_calls,
    _sanitize_agent_leak,
    _StreamToolCallSplitter,
)

FAILED = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        FAILED.append(name)


# ---------------------------------------------------------------------------
# Case 1: deepseek-v4-flash 伪回调头 <tool_callback>（不闭合）+ 正常 JSON 调用
# 取自 jsonl line 252 的原始输出
# ---------------------------------------------------------------------------
deepseek_out = (
    "<tool_callback>\n"
    "用几本跟《打开心智》风格最接近的书来查，一起拉出来。\n\n"
    "<tool_call>\n"
    '{"name": "shell", "arguments": {"command": "echo hi", '
    '"intent": "批量查同类书详情", "timeout": 60}}\n'
    "</tool_call>"
)

rest, calls = _parse_tool_calls(deepseek_out)
check("case1 解析出 shell 调用", len(calls) == 1 and calls[0]["function"]["name"] == "shell")
check("case1 tool_callback 不泄漏进正文", "tool_callback" not in rest and "用几本" not in rest,
      f"rest={rest!r}")

# 流式路径：splitter 必须从 <tool_callback> 起扣留，正文 chunk 里不能出现
sp = _StreamToolCallSplitter(frozenset({"shell"}))
streamed = sp.feed(deepseek_out[:20]) + sp.feed(deepseek_out[20:60])
streamed += sp.feed(deepseek_out[60:])
tail, scalls = sp.flush()
tail2, scalls2 = _parse_tool_calls(tail, frozenset({"shell"}))
check("case1 流式正文无 tool_callback 泄漏", "tool_callback" not in streamed + tail2,
      f"streamed={streamed!r} tail={tail2!r}")
check("case1 流式解析出调用", len(scalls) == 1 and scalls[0]["function"]["name"] == "shell")

# 非对话（cleaner）路径
cleaned = _sanitize_agent_leak(deepseek_out)
check("case1 sanitize 后无残留", "tool_callback" not in cleaned and "用几本" not in cleaned,
      f"cleaned={cleaned!r}")

# ---------------------------------------------------------------------------
# Case 2: glm-5.3（请求模型名 claude-opus-4-7）函数调用语法 + 孤立 </arg_value>
# 取自 jsonl line 212 的原始输出
# ---------------------------------------------------------------------------
glm_out = '<tool_call>skill_read(intent="读book-info技能用法")</arg_value></tool_call>'
rest, calls = _parse_tool_calls(glm_out)
check("case2 解析出 skill_read 调用", len(calls) == 1 and calls[0]["function"]["name"] == "skill_read")
if calls:
    args = __import__("json").loads(calls[0]["function"]["arguments"])
    check("case2 intent 参数保留", args.get("intent") == "读book-info技能用法", f"args={args}")
check("case2 正文无残骸", "arg_value" not in rest and "skill_read" not in rest, f"rest={rest!r}")

cleaned = _sanitize_agent_leak('<tool_call>skill_read(intent="x")</arg_value></tool_call>tail')
check("case2 sanitize 路径剥 arg_value 残骸", "arg_value" not in cleaned, f"cleaned={cleaned!r}")

# ---------------------------------------------------------------------------
# Case 3: 回归——教学格式（JSON）不受影响
# ---------------------------------------------------------------------------
normal = '<tool_call>\n{"name": "shell", "arguments": {"command": "ls"}}\n</tool_call>'
rest, calls = _parse_tool_calls(normal)
check("case3 教学格式仍正常", len(calls) == 1 and rest == "")

# 复数变体 + 多 JSON 回归
plural = '<tool_calls>\n{"name": "a1", "arguments": {}}\n{"name": "a2", "arguments": {}}\n</tool_calls>'
rest, calls = _parse_tool_calls(plural)
check("case3 复数变体多调用回归", len(calls) == 2)

# 裸 JSON 回归
bare = '好的。\n{"name": "shell", "arguments": {"command": "ls"}}'
rest, calls = _parse_tool_calls(bare, frozenset({"shell"}))
check("case3 裸 JSON 回归", len(calls) == 1 and rest == "好的。")

# XML 属性风格回归
attr_style = '<tool_call name="shell" command="ls" intent="列目录" />'
rest, calls = _parse_tool_calls(attr_style)
check("case3 属性风格回归", len(calls) == 1 and calls[0]["function"]["name"] == "shell")

# 正文里的 tool_callback 伪头 + 前置正文（不能误删前面的正常文本）
prefix_case = (
    "我先查一下。\n"
    "<tool_callback>\n查同类书\n\n"
    "<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\": \"ls\"}}\n</tool_call>"
)
rest, calls = _parse_tool_calls(prefix_case)
check("case3 前置正文保留", rest.startswith("我先查一下。"), f"rest={rest!r}")
check("case3 伪头剥离", "tool_callback" not in rest and "查同类书" not in rest, f"rest={rest!r}")
check("case3 调用解析", len(calls) == 1)

# 已闭合的 tool_callback 块也能整块剥离
closed_cb = "<tool_callback>\n意图文本\n</tool_callback>\n<tool_call>\n{\"name\": \"shell\", \"arguments\": {}}\n</tool_call>"
rest, calls = _parse_tool_calls(closed_cb)
check("case3 闭合 tool_callback 整块剥离", "意图文本" not in rest and len(calls) == 1, f"rest={rest!r}")

# ---------------------------------------------------------------------------
# Case 4: 未闭合 tool_callback 且放弃调用（模型继续说正文）——
# 只剥标签壳，伪头之后的正文必须保留（PR #13 评审意见）
# ---------------------------------------------------------------------------
abandon = "<tool_callback>\n用几本同类的书来查\n\n还是直接回答吧：这几本书都不错。"
rest, calls = _parse_tool_calls(abandon)
check("case4 正文保留", "还是直接回答吧" in rest, f"rest={rest!r}")
check("case4 标签壳剥离", "tool_callback" not in rest, f"rest={rest!r}")
cleaned = _sanitize_agent_leak(abandon)
check("case4 sanitize 正文保留", "还是直接回答吧" in cleaned and "tool_callback" not in cleaned,
      f"cleaned={cleaned!r}")

# ---------------------------------------------------------------------------
# Case 5: glm-5.3 Scopus 会话变体——<tool_call> 块内连续多个函数调用表达式
# + 孤立 </arg_value> 残骸（用户实测粘贴，ethan 会话 s_20260903_1704_0492）
# ---------------------------------------------------------------------------
scopus_wrapped = (
    '<tool_call>'
    'web_search(query="Scopus database indexed journals count 2025 2026 coverage", '
    'max_results=7, language="en-US", intent="核实Scopus收录规模数据") '
    'web_search(query="Scopus arXiv preprints indexing 预印本 收录", '
    'max_results=7, language="zh-CN", intent="核实Scopus是否收录arXiv预印本")'
    '</arg_value></tool_call>'
)
rest, calls = _parse_tool_calls(scopus_wrapped)
check("case5 解析出 2 个 web_search 调用",
      len(calls) == 2 and all(c["function"]["name"] == "web_search" for c in calls),
      f"n={len(calls)} rest={rest!r}")
if len(calls) == 2:
    a0 = __import__("json").loads(calls[0]["function"]["arguments"])
    a1 = __import__("json").loads(calls[1]["function"]["arguments"])
    check("case5 引号参数保留", a0.get("query") == "Scopus database indexed journals count 2025 2026 coverage")
    check("case5 未加引号数字参数不丢", a0.get("max_results") == 7, f"a0={a0}")
    check("case5 两调用参数独立", a1.get("language") == "zh-CN" and a1.get("max_results") == 7)
check("case5 正文无残骸", "web_search" not in rest and "arg_value" not in rest, f"rest={rest!r}")

# ---------------------------------------------------------------------------
# Case 6: 同源变体——完全无包裹的裸函数调用表达式（残骸伴随）
# ---------------------------------------------------------------------------
scopus_bare = (
    'web_search(query="Scopus database indexed journals count 2025 2026 coverage", '
    'max_results=7, language="en-US", intent="核实Scopus收录规模数据") '
    'web_search(query="Scopus arXiv preprints indexing 预印本 收录", '
    'max_results=7, language="zh-CN", intent="核实Scopus是否收录arXiv预印本")'
    '</arg_value>'
)
rest, calls = _parse_tool_calls(scopus_bare, frozenset({"web_search", "web_fetch"}))
check("case6 裸表达式解析出 2 个调用",
      len(calls) == 2 and all(c["function"]["name"] == "web_search" for c in calls),
      f"n={len(calls)} rest={rest!r}")
check("case6 正文无残骸", "web_search" not in rest and "arg_value" not in rest, f"rest={rest!r}")

# 未知工具名的裸表达式不动（防误伤）
prose_case = 'foo(query="x") bar(y=1)'
rest, calls = _parse_tool_calls(prose_case, frozenset({"web_search"}))
check("case6 未知工具名不转换", len(calls) == 0 and "foo(query" in rest, f"rest={rest!r}")

# 无残骸时，行中 inline 的已知工具名表达式不转换（可能是正文代码示例）
inline_prose = '你可以用 web_search(query="x") 来搜索，注意参数写法。'
rest, calls = _parse_tool_calls(inline_prose, frozenset({"web_search"}))
check("case6 inline 示例不转换", len(calls) == 0 and "web_search" in rest, f"rest={rest!r}")

# ---------------------------------------------------------------------------
# Case 7: 流式路径——裸函数调用表达式跨 chunk 到达，splitter 必须扣留并解析
# ---------------------------------------------------------------------------
sp = _StreamToolCallSplitter(frozenset({"web_search", "web_fetch"}))
streamed = ""
for k in range(0, len(scopus_bare), 25):
    streamed += sp.feed(scopus_bare[k:k + 25])
tail, scalls = sp.flush()
tail2, scalls2 = _parse_tool_calls(tail, frozenset({"web_search", "web_fetch"}))
check("case7 流式正文无泄漏", "web_search" not in streamed + tail2, f"streamed={streamed!r} tail={tail2!r}")
check("case7 流式解析出调用", len(scalls) + len(scalls2) == 2,
      f"scalls={len(scalls)} scalls2={len(scalls2)}")

# 带包裹的变体流式也要扣住
sp2 = _StreamToolCallSplitter(frozenset({"web_search"}))
streamed2 = ""
for k in range(0, len(scopus_wrapped), 30):
    streamed2 += sp2.feed(scopus_wrapped[k:k + 30])
tail3, scalls3 = sp2.flush()
check("case7 包裹变体流式无泄漏", "web_search" not in streamed2 + tail3, f"{(streamed2 + tail3)!r}")
check("case7 包裹变体流式解析", len(scalls3) == 2, f"n={len(scalls3)}")

# 正文行首恰好像调用的普通文本（无残骸）最终要放行，不能吞正文
prose_stream = "web_search 是搜索工具，不是试错工具。"
sp3 = _StreamToolCallSplitter(frozenset({"web_search"}))
streamed3 = sp3.feed(prose_stream)
tail4, _ = sp3.flush()
check("case7 普通正文最终放行", "是搜索工具" in (streamed3 + tail4),
      f"streamed={streamed3!r} tail={tail4!r}")

print()
if FAILED:
    print("FAILED:", FAILED)
    sys.exit(1)
print("ALL PASS")
