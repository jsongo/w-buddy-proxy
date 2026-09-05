"""上游 agent 预设泄漏防护：压制指令注入 + 正文/流式清洗。

Trae 的 llm_utils_chat（solo_work_lite 等预设包装）会注入"带 shell 工具的
coding agent"预设，纯对话场景模型会把 Command 块等工具语法当正文吐出。
本模块负责剥离这些协议残留。原生 function calling 通道（native_tools）无
agent 预设，不需要这层防护。
"""

from __future__ import annotations

import logging
import re
from typing import Any

log = logging.getLogger(__name__)

# 注入的 agent 协议压制指令（见 _extract_prompt 说明）
_AGENT_GUARD = (
    "当前环境是纯文本对话 API，没有任何命令执行、文件读写等工具，也没有 shell。"
    "禁止输出任何工具调用标记（如 <Command>、<tool_action>、execute_command 等"
    "XML 标签或伪协议块）。如问题看似需要执行命令/操作系统，直接用自然语言给出"
    "回答或方案，示例命令用 markdown 代码块展示。"
)

# 输出侧兜底清洗：即使 system 注入失效，也把漏出的 agent 协议标记剥掉。
# 实测泄漏格式（Trae agent 端点的服务端预设导致模型输出工具调用语法）：
#   <Command len=22 id=0 run=1369463>\nlsof ...\n</Command>
#   以及 tool_call 残留前缀、tool_action 块、think 标签、错误提示行。
# 注意 <Command\\s 要求标签名后跟空白接属性（len=/id=/run=），避免误伤
# <CommandBuffer> 这类正常技术词汇；正文中单独的 Command 单词不会匹配。
_LEAK_RE = [
    # 带属性签名的命令块（len=/id=，可带 tool_call 残留前缀；
    # 上游泄漏格式可能残缺：< 不总存在、截断流可能没有 >）
    re.compile(r"(?:</?\s*tool_call>\s*)?<?\s*Command\s+len=\d+\s+id=\d+[^>\n]*>?.*?</Command>", re.S),
    re.compile(r"(?:</?\s*tool_call>\s*)?<?\s*Command\s+len=\d+\s+id=\d+[^>\n]*>?.*", re.S),  # 未闭合尾巴（截断流）
    # 常规标签形态的命令块（backup）
    re.compile(r"<Command\s[^>]*>.*?</Command>", re.S),
    re.compile(r"<Command\s[^>]*>.*", re.S),
    re.compile(r"<Command\s[^>]*>"),
    re.compile(r"</Command>"),
    # tool_action 完整块与孤立标签
    re.compile(r"<tool_action[^>]*>.*?</tool_action>", re.S),
    re.compile(r"</?\s*tool_action[^>]*>"),
    # 孤立的 tool_name / command 块（tool_action 外壳被剥离/缺失时，
    # 内部标签会裸露——实测下游 coding agent 收到的正是这种残骸）
    re.compile(r"<tool_name[^>]*>.*?</tool_name>", re.S),
    re.compile(r"</?\s*tool_name[^>]*>"),
    re.compile(r"<command>.*?</command>", re.S | re.I),
    re.compile(r"</?\s*command>", re.I),
    # think 块（reasoning 泄漏到正文）与孤立 think 标签
    re.compile(r"<think>.*?</think>", re.S),
    re.compile(r"</?\s*think>"),
    # 伪回调头 <tool_callback>（deepseek-v4-flash 实测：开标签后跟意图文本、
    # 不闭合也不携带调用，纯粹是协议噪音）。剥法与 _parse_tool_calls 统一：
    # 闭合块整块剥；未闭合但后面跟 <tool_call> 的，剥到调用前（真调用交给
    # 后续/解析层）；真到流尾都没闭合的（模型放弃调用继续说正文），只剥
    # 标签壳、保留内部文本——否则伪头之后的正文会被整段吞掉。
    # 必须排在孤立 tool_call 模式之前——tool_call[^>]* 会把 <tool_callback>
    # 误当孤儿 tool_call 标签先剥壳，导致块模式失配、内部文本泄漏
    re.compile(r"<tool_callback[^>]*>.*?</tool_callback>", re.S),
    re.compile(r"<tool_callback[^>]*>.*?(?=<tool_call[>\s])", re.S),
    re.compile(r"</?\s*tool_callback[^>]*>"),
    # 孤立 tool_call 标签
    re.compile(r"</?\s*tool_call[^>]*>"),
    # 孤立 arg_key / arg_value 标签（工具调用残骸，glm-5.3 实测）
    re.compile(r"</?\s*arg_(?:key|value)[^>]*>"),
    # seed 协议块（doubao-seed-evolving 实测 s_20260905_0233_f291）：
    # <seed:tool_call><function name="..."><parameter ...>...</parameter></function></seed:tool_call>
    re.compile(r"<seed:tool_call>[\s\S]*?(?:</seed:tool_call>|\Z)", re.I),
    re.compile(r"</?\s*seed:tool_call[^>]*>", re.I),
    re.compile(r"<function\s+name=[^>]*>[\s\S]*?</function>", re.I),
    re.compile(r"</?\s*function[^>]*>", re.I),
    re.compile(r"</?\s*parameter[^>]*>", re.I),
    # 工具执行失败的提示行
    re.compile(r"执行过程发生错误[:：][^\n]*"),
]


def _sanitize_agent_leak(text: str) -> str:
    """剥离上游 agent 协议残留（Command 块 / tool_call / tool_action 标签等）。

    仅用于纯对话请求（_looks_like_agent_request 判定为 False）：coding agent
    请求需要保留模型输出的工具调用语法（下游自己解析），不能走本函数。
    """
    if not any(k in text for k in (
        "Command", "tool_call", "tool_action", "tool_name",
        "tool_callback", "<command", "arg_value", "arg_key",
        "think", "执行过程发生错误",
        "seed:tool_call", "<function", "<parameter",
    )):
        return text
    for pat in _LEAK_RE:
        text = pat.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# 泄漏标签名（流式扣留判断用；大小写不敏感）
_LEAK_TAG_NAMES = ("tool_action", "tool_name", "tool_call", "tool_callback",
                   "think", "command", "seed:tool_call", "function", "parameter")


def _stream_hold_pos(buf: str, tags: tuple[str, ...] = _LEAK_TAG_NAMES) -> int:
    """返回应扣留的起始位置（-1 表示整段可安全释放）。

    两种扣留场景：
    1. 存在未闭合的标签开启（如 <tool_action> 还没等到 </tool_action>）
       ——必须等闭合标签到齐后整块处理，否则会出现"剥了外壳留下内脏"
       的残骸（下游收到的正是这种，解析器不认）
    2. 尾部的裸 "<" 是某个标签名的前缀（如 "<tool_ac"）——扣住防止
       标签名被 SSE 分包分裂
    """
    low = buf.lower()
    best = -1
    for tag in tags:
        start = 0
        opener = f"<{tag}"
        while True:
            i = low.find(opener, start)
            if i == -1:
                break
            after = low[i + len(opener): i + len(opener) + 1]
            if after in ("", ">", "/", " ", "\t", "\n"):
                closer = f"</{tag}>"
                if low.find(closer, i) == -1 and (best == -1 or i < best):
                    best = i
            start = i + 1
    if best != -1:
        return best
    j = buf.rfind("<")
    if j != -1 and ">" not in buf[j:]:
        partial = buf[j + 1:].lower().lstrip("/")
        if any(t.startswith(partial) for t in _LEAK_TAG_NAMES):
            return j
    return -1


class _StreamLeakCleaner:
    """流式泄漏清洗器：跨 chunk 缓冲 + 整块清洗。

    之前的实现按 chunk 逐段调 _sanitize_agent_leak，标签跨 SSE 分包分裂时
    整块正则匹配不上、孤儿标签正则只剥掉首尾外壳，内部内容反而漏出去。
    本清洗器把"可能不完整"的尾部扣在缓冲区，只释放确定安全的前缀。
    """

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, text: str) -> str:
        self._buf += text
        pos = _stream_hold_pos(self._buf)
        if pos == -1:
            safe, self._buf = self._buf, ""
        else:
            safe, self._buf = self._buf[:pos], self._buf[pos:]
        return _sanitize_agent_leak(safe)

    def flush(self) -> str:
        out = _sanitize_agent_leak(self._buf)
        self._buf = ""
        return out

