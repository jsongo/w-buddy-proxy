"""Bare JSON tool call variant tests."""
import json
import sys

sys.path.insert(0, "src")
from codebuddy_proxy.trae_provider import (  # noqa: E402
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

# 5. 安全性：杂键 → 不误伤
c5 = '{"name": "shell", "arguments": {"command": "ls"}, "extra": 1}'
rest, calls = _parse_tool_calls(c5, KNOWN)
check("裸JSON: 杂键不误伤", len(calls) == 0 and "extra" in rest)

# 6. 安全性：缺 arguments → 不误伤
c6 = '{"name": "shell"}'
rest, calls = _parse_tool_calls(c6, KNOWN)
check("裸JSON: 缺arguments不误伤", len(calls) == 0)

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

# 18. XML 属性：未闭合标签保留原文（流式截断 flush 场景）
cx3 = '文字 \x3ctool_call name="shell" command="未完结'
rest, calls = _parse_tool_calls(cx3, KNOWN)
check("XML属性: 未闭合保留", len(calls) == 0 and "未完结" in rest)

# 19. XML 属性：流式跨 chunk 分裂
sp = _StreamToolCallSplitter(KNOWN)
out = []
for p in ['查一下。\n', '\x3ctool_call na', 'me="shell" comm', 'and="pwd" intent="查" /\x3e']:
    out.append(sp.feed(p))
tail, calls = sp.flush()
check("XML属性: 流式跨chunk", "".join(out) == "查一下。\n" and len(calls) == 1)

print()
print("RESULT:", "ALL PASS" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
