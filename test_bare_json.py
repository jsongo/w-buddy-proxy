"""Bare JSON tool call variant tests."""
import json
import sys

sys.path.insert(0, "src")
from buddy_proxy.trae_provider import (  # noqa: E402
    _parse_tool_calls,
    _StreamToolCallSplitter,
    _TC_OPEN,
    _TC_CLOSE,
    _tool_names,
)

O, C = _TC_OPEN, _TC_CLOSE
KNOWN = frozenset({"shell", "file_read", "recall_memory"})

fails = []


def check(name, cond, extra=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {extra}")
    if not cond:
        fails.append(name)


# 1. 实测泄漏：行首裸 JSON（单个）
c = 'auth.json 里已经有 key 了。让我直接写进去。\n\n{"name": "shell", "arguments": {"command": "python3 -c 1"}, "intent": "写配置"}\n'
rest, calls = _parse_tool_calls(c, KNOWN)
check("裸JSON: 单个提取", len(calls) == 1 and calls[0]["function"]["name"] == "shell"
      and "写进去" in rest and "shell" not in rest, f"rest={rest!r}")

# 2. 实测泄漏：连续两个裸 JSON
c2 = ('{"name": "shell", "arguments": {"command": "ethan provider list"}, "intent": "查配置"}\n'
      '{"name": "shell", "arguments": {"command": "ls ~/.pi/"}, "intent": "查目录"}\n')
rest, calls = _parse_tool_calls(c2, KNOWN)
check("裸JSON: 连续两个", len(calls) == 2 and rest == "", f"rest={rest!r}")

# 3. 数组包裹：[{"name": ...}]
c3 = '我来看下。\n[{"name": "shell", "arguments": {"command": "ls"}}, {"name": "file_read", "arguments": {"path": "/etc/hosts"}}]\n'
rest, calls = _parse_tool_calls(c3, KNOWN)
check("裸JSON: 数组包裹两个", len(calls) == 2
      and calls[0]["function"]["name"] == "shell" and calls[1]["function"]["name"] == "file_read",
      f"rest={rest!r}")

# 4. 安全性：name 不在工具列表 → 不误伤
c4 = '配置如下：\n{"name": "deepseek-chat", "arguments": {"model": "v3"}, "id": "x"}\n继续。'
rest, calls = _parse_tool_calls(c4, KNOWN)
check("裸JSON: 未知name不误伤", len(calls) == 0 and "deepseek-chat" in rest, f"rest={rest!r}")

# 5. 杂键：行首 + 已知工具名 + 完整对象 → 闸门补收为调用（旧行为是
#    保留为正文，实际是泄漏；杂键被忽略）
c5 = '{"name": "shell", "arguments": {"command": "ls"}, "extra": 1}'
rest, calls = _parse_tool_calls(c5, KNOWN)
check("裸JSON: 杂键由闸门补收", len(calls) == 1 and rest == ""
      and json.loads(calls[0]["function"]["arguments"])["command"] == "ls",
      f"calls={len(calls)} rest={rest!r}")

# 6. 裸名字 {"name": "shell"}：行首 + 已知工具 → 闸门补收为空参调用
c6 = '{"name": "shell"}'
rest, calls = _parse_tool_calls(c6, KNOWN)
check("裸JSON: 裸名字闸门补收", len(calls) == 1 and rest == ""
      and json.loads(calls[0]["function"]["arguments"]) == {},
      f"calls={len(calls)} rest={rest!r}")

# 7. 安全性：不传 known_tools → 完全不启用裸解析
c7 = '{"name": "shell", "arguments": {"command": "ls"}}'
rest, calls = _parse_tool_calls(c7)
check("裸JSON: 无known_tools不启用", len(calls) == 0 and c7 in rest)

# 8. 标签格式存在时优先，不跑裸解析（不双算）
c8 = f'好。\n{O}\n{{"name": "shell", "arguments": {{"command": "ls"}}}}\n{C}\n'
rest, calls = _parse_tool_calls(c8, KNOWN)
check("裸JSON: 标签格式优先", len(calls) == 1)

# 9. 缩进的裸 JSON 也识别
c9 = '看看：\n  {"name": "shell", "arguments": {"command": "pwd"}}\n'
rest, calls = _parse_tool_calls(c9, KNOWN)
check("裸JSON: 缩进识别", len(calls) == 1)

# 10. 普通含 < 的文本 + JSON 数组正文不误伤
c10 = '一些配置示例：\n{"name": "server", "arguments": {"port": 8080}}\n完。'
rest, calls = _parse_tool_calls(c10, KNOWN)
check("裸JSON: 普通JSON正文保留", len(calls) == 0 and "port" in rest)

# 11. 流式：裸 JSON 跨 chunk 分裂
sp = _StreamToolCallSplitter(KNOWN)
out = []
pieces = ['让我查一下。\n\n', '{"na', 'me": "sh', 'ell", "argu', 'ments": {"command": "ls -la"}, "intent": "查"}',
          '\n', '{"name": "file_read", "arg', 'uments": {"path": "/tmp"}}\n']
for p in pieces:
    out.append(sp.feed(p))
tail, calls = sp.flush()
check("流式: 裸JSON跨chunk", "".join(out) == "让我查一下。\n\n" and len(calls) == 2,
      f"out={''.join(out)!r} calls={len(calls)}")

# 12. 流式：正文里 JSON 数组（未知工具）正常流出
sp = _StreamToolCallSplitter(KNOWN)
out = []
for p in ['配置：\n', '{"name": "models", "arguments": {"x": 1}}', '\n完了。']:
    out.append(sp.feed(p))
tail, calls = sp.flush()
check("流式: 未知name正常流出", len(calls) == 0 and "models" in "".join(out) + tail)

# 13. 流式：无 known_tools 时行为与旧版一致
sp = _StreamToolCallSplitter()
out = [sp.feed('正文 {"name": "shell", "arguments": {}} 流过')]
tail, calls = sp.flush()
check("流式: 无tools不扣留", len(calls) == 0 and "shell" in "".join(out) + tail)

# 14. _tool_names 提取
tools = [{"type": "function", "function": {"name": "a"}},
         {"type": "function", "function": {"name": "b"}}, "bad"]
names = _tool_names(tools)
check("_tool_names", names == frozenset({"a", "b"}))

# 15. 教学格式混搭回归（上次的修复不破坏）
mixed = f'文字\n{O}\n{{"name": "shell", "arguments": {{"command": "ls"}}}}\n</tool_calls>\n'
rest, calls = _parse_tool_calls(mixed, KNOWN)
check("回归: 混搭标签", len(calls) == 1 and "文字" in rest)

# 16. XML 属性风格（实测 deepseek-v4-pro 偶发变体，含 > 的命令）
TAG = "\x3ctool_call"
cx = ('让我重新查一下，之前的查询结果没回来。\n\n'
      + TAG + ' name="shell" command="ls ~/.pi/ 2>/dev/null && echo ok" intent="查 pi agent 配置目录" /\x3e\n')
rest, calls = _parse_tool_calls(cx, KNOWN)
argsx = json.loads(calls[0]["function"]["arguments"]) if calls else {}
check("XML属性: 含>命令提取", len(calls) == 1 and calls[0]["function"]["name"] == "shell"
      and argsx.get("command") == "ls ~/.pi/ 2>/dev/null && echo ok"
      and argsx.get("intent") == "查 pi agent 配置目录" and "tool_call" not in rest,
      f"rest={rest!r}")

# 17. XML 属性：JSON arguments + 单引号值
cx2 = "\x3ctool_action name='shell' command='ls -la' /\x3e"
rest, calls = _parse_tool_calls(cx2, KNOWN)
check("XML属性: 单引号", len(calls) == 1)

# 18. XML 属性：未闭合标签随闸门清理丢弃（2026-09-05 起：损坏标签壳属
#     协议残骸，连属性值一并清掉，不再保留为正文）
cx3 = '文字 \x3ctool_call name="shell" command="未完结'
rest, calls = _parse_tool_calls(cx3, KNOWN)
check("XML属性: 未闭合清理", len(calls) == 0 and "未完结" not in rest
      and "tool_call" not in rest and "文字" in rest, f"rest={rest!r}")

# 19. XML 属性：流式跨 chunk 分裂
sp = _StreamToolCallSplitter(KNOWN)
out = []
for p in ['查一下。\n', '\x3ctool_call na', 'me="shell" comm', 'and="pwd" intent="查" /\x3e']:
    out.append(sp.feed(p))
tail, calls = sp.flush()
check("XML属性: 流式跨chunk", "".join(out) == "查一下。\n" and len(calls) == 1)

# 20. 未转义双引号修复：单个裸 JSON（deepseek-v4-pro 实测 echo "---"）
bad_q = '{ "name": "shell", "arguments": { "command": "ls 2>/dev/null; echo "---"; ls | grep diff", "intent": "查diff" } }'
rest, calls = _parse_tool_calls(bad_q, KNOWN)
check("引号修复: 单个裸JSON", len(calls) == 1 and calls[0]["function"]["name"] == "shell")
if calls:
    aq = json.loads(calls[0]["function"]["arguments"])
    check("引号修复: 命令保留", 'echo "---"' in aq.get("command", ""))

# 21. 未转义双引号修复：{ 和 "name" 之间有空格
bad_sp = '{ "name": "shell", "arguments": { "command": "echo "hi"", "intent": "test" } }'
rest, calls = _parse_tool_calls(bad_sp, KNOWN)
check("引号修复: 空格形态", len(calls) == 1)

# 22. 未转义双引号修复：同行连续多个
bad_multi = (
    '{"name": "shell", "arguments": {"command": "echo "PR1" && ls", "intent": "PR1"}} '
    '{"name": "shell", "arguments": {"command": "echo "PR2" && ls", "intent": "PR2"}}'
)
rest, calls = _parse_tool_calls(bad_multi, KNOWN)
check("引号修复: 同行连续2个", len(calls) == 2, f"calls={len(calls)}")

# 23. 未转义双引号修复：<tool_call> 包裹
bad_wrapped = '<tool_call>\n{"name": "shell", "arguments": {"command": "echo "hello" && ls"}}\n</tool_call>'
rest, calls = _parse_tool_calls(bad_wrapped, KNOWN)
check("引号修复: tool_call包裹", len(calls) == 1)

# 24. 未转义双引号修复：流式路径
sp = _StreamToolCallSplitter(KNOWN)
out = []
full = '查一下。\n' + bad_q
for k in range(0, len(full), 25):
    out.append(sp.feed(full[k:k+25]))
tail, calls = sp.flush()
tail2, calls2 = _parse_tool_calls(tail, KNOWN)
check("引号修复: 流式", len(calls) + len(calls2) == 1 and "shell" not in "".join(out))

# 25. 正常 JSON 不被误修
good = '{"name": "shell", "arguments": {"command": "ls -la", "intent": "列目录"}}'
rest, calls = _parse_tool_calls(good, KNOWN)
check("引号修复: 正常JSON不变", len(calls) == 1 and rest == "")
if calls:
    ag = json.loads(calls[0]["function"]["arguments"])
    check("引号修复: 正常参数", ag["command"] == "ls -la")

# 26. 值里含 } （echo "}"）：扫描器跟踪字符串状态，边界不能提前归零
brace_v = '{ "name": "shell", "arguments": { "command": "echo "}" 2>/dev/null; ls", "intent": "查" } }'
rest, calls = _parse_tool_calls(brace_v, KNOWN)
check("引号修复: 值含}解析出调用", len(calls) == 1, f"calls={len(calls)} rest={rest!r}")
if calls:
    ab = json.loads(calls[0]["function"]["arguments"])
    check("引号修复: 值含}命令完整", ab.get("command") == 'echo "}" 2>/dev/null; ls', f"args={ab}")

# 27. 尾部截断（流断在字符串中间）：截断修复补闭合引号 + 括号，打捞成调用
#     （旧行为是整段保留为正文 = 泄漏）
broken = '{"name": "shell", "arguments": {"command": "ls 未闭合'
rest, calls = _parse_tool_calls(broken, KNOWN)
check("截断修复: 字符串内截断打捞", len(calls) == 1
      and calls[0]["function"]["name"] == "shell" and rest == "",
      f"calls={len(calls)} rest={rest!r}")
if calls:
    ab = json.loads(calls[0]["function"]["arguments"])
    check("截断修复: 半截参数保留", ab.get("command") == "ls 未闭合", f"args={ab}")

# 28. 冒号连写形态（glm-5.3 实测）：web_searchquery: <自由文本>，
#     连续两个时上一个值与下一个标记粘连，尾随 markdown --- 分隔线
WS = frozenset({"web_search", "web_fetch"})
colon = ("web_searchquery: how to identify if underlying model is Claude "
         "fingerprint methodsweb_searchquery: 检测套壳模型是否 Claude 方法 提示词探针\n---")
rest, calls = _parse_tool_calls(colon, WS)
check("冒号连写: 解析2个", len(calls) == 2, f"n={len(calls)} rest={rest!r}")
if len(calls) == 2:
    c0 = json.loads(calls[0]["function"]["arguments"])
    c1 = json.loads(calls[1]["function"]["arguments"])
    check("冒号连写: q1完整", c0.get("query", "").endswith("methods") and "how to identify" in c0["query"])
    check("冒号连写: q2完整", c1.get("query") == "检测套壳模型是否 Claude 方法 提示词探针")
check("冒号连写: 无残骸", "web_search" not in rest and "---" not in rest, f"rest={rest!r}")

# 29. 冒号连写：单个 + 前导正文
colon2 = "我来搜一下。\nweb_searchquery: claude fingerprint 检测方法"
rest, calls = _parse_tool_calls(colon2, WS)
check("冒号连写: 前导正文", len(calls) == 1 and "我来搜" in rest, f"n={len(calls)} rest={rest!r}")

# 30. 冒号连写：流式跨 chunk
sp = _StreamToolCallSplitter(WS)
out = []
for k in range(0, len(colon), 20):
    out.append(sp.feed(colon[k:k + 20]))
tail, calls = sp.flush()
t2, calls2 = _parse_tool_calls(tail, WS)
check("冒号连写: 流式无泄漏", len(calls) + len(calls2) == 2
      and "web_search" not in "".join(out), f"n={len(calls) + len(calls2)} out={''.join(out)!r}")

# 31. 冒号连写：普通正文（工具名后有空格再跟冒号）不误伤
prose = "web_search 是一个工具: query 参数必填。"
rest, calls = _parse_tool_calls(prose, WS)
check("冒号连写: 正文不误伤", len(calls) == 0 and "工具" in rest, f"rest={rest!r}")

print()
print("RESULT:", "ALL PASS" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
