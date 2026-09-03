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

print()
if FAILED:
    print("FAILED:", FAILED)
    sys.exit(1)
print("ALL PASS")
