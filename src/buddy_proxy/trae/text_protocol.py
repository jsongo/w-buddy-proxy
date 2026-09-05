"""文本协议请求侧转换：工具教学指令、agent 请求识别、OpenAI messages -> Trae。

原生通道（native_tools）的上游回退路径。_extract_prompt 把 OpenAI 消息转成
Trae block 数组，注入 guard/工具教学，序列化历史 tool_calls 为调用块文本。
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

from .leak_guard import _AGENT_GUARD
from .text_toolcall import _TC_CLOSE, _TC_OPEN

log = logging.getLogger(__name__)

def _looks_like_agent_request(messages: list[dict[str, Any]], body: dict[str, Any]) -> bool:
    """判断请求是否来自自带工具协议的下游（coding agent / function calling）。

    这类请求里模型输出的工具调用语法（<tool_action> 等）是下游 agent 自己
    要解析的，不能注入压制指令、也不能清洗（实测模型会完美遵循下游 system
    prompt 里教的格式；反而我们的 guard 注入会让模型输出两段指令打架的
    纠结文本，逐 chunk 清洗会把 tool_action 外壳剥掉留下内脏残骸）。

    判定信号（任一命中即视为 agent 请求）：
    1. 请求带 tools / tool_choice 字段（标准 function calling）
    2. messages 里有 role=tool/function 的消息（工具结果回传）
    3. system 消息里有工具语法教学（<tool_action / <tool_name / tool_call /
       execute_command 等）
    4. 历史消息正文里出现过完整的工具调用块（agent 循环中的多轮请求）
    """
    if body.get("tools") or body.get("tool_choice") is not None:
        return True
    for m in messages:
        role = m.get("role", "")
        if role in ("tool", "function"):
            return True
        content = m.get("content", "")
        if isinstance(content, list):
            content = "".join(
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            )
        if not isinstance(content, str) or not content:
            continue
        low = content.lower()
        if role == "system":
            if any(k in low for k in (
                "<tool_action", "<tool_name", "tool_call", "<function_call",
                "execute_command", "tool use", "工具调用", "你可以调用工具",
            )):
                return True
        else:
            # 非系统消息只认强信号：出现过完整的工具调用块
            if "<tool_action" in low and "</tool_action>" in low:
                return True
    return False

def _build_tools_system(tools: list[dict[str, Any]]) -> str:
    """把 OpenAI tools 定义序列化成 prompt-based function calling 教学指令。

    Trae 上游不支持原生 tools 参数，用提示词教学 + 输出解析的方式模拟。
    """
    defs = []
    for t in tools:
        f = t.get("function") if isinstance(t, dict) else None
        if not isinstance(f, dict) or not f.get("name"):
            continue
        defs.append({
            "name": f.get("name", ""),
            "description": f.get("description", ""),
            "parameters": f.get("parameters", {}),
        })
    if not defs:
        return ""
    return (
        "你可以调用下列工具（function calling）。工具列表：\n"
        + json.dumps(defs, ensure_ascii=False)
        + "\n\n调用规则：\n"
        "- 需要调用工具时，输出如下格式的调用块（JSON 一行，可连续多个块）：\n"
        + _TC_OPEN + '\n{"name": "工具名", "arguments": {参数对象}}\n' + _TC_CLOSE + "\n"
        "- arguments 必须是合法 JSON 对象，与工具参数 schema 一致。\n"
        "- 每次调用都必须带完整的开闭标签，即使是同一任务里的第 N 次调用；"
        "禁止省略标签直接输出裸 JSON。\n"
        "- 调用块外可以写简短的说明文字；不要把工具参数写进正文；"
        "不要发明列表外的工具。\n"
        "- 收到 [tool_result] 开头的消息后，那是工具执行结果，据此继续任务，"
        "直到可以给出最终回答。如果对话里还没有出现 [tool_result]，"
        "说明工具从未执行过——不要假设命令已运行、不要等待结果，"
        "需要结果就直接再输出调用块。\n"
    )


def _serialize_tool_calls(tool_calls: list[dict[str, Any]]) -> str:
    """把历史 assistant.tool_calls 序列化成教学格式的调用块文本。"""
    parts = []
    for c in tool_calls:
        f = c.get("function") or {}
        try:
            args = json.loads(f.get("arguments") or "{}")
        except Exception:
            args = {"raw": f.get("arguments", "")}
        parts.append(_TC_OPEN + "\n" + json.dumps(
            {"name": f.get("name", ""), "arguments": args}, ensure_ascii=False
        ) + "\n" + _TC_CLOSE)
    return "\n".join(parts)
def _content_blocks(content: Any) -> list[dict[str, Any]]:
    """OpenAI content -> Trae block 数组，保留 image_url 等多模态 block。

    上游 llm_utils_chat 认 OpenAI 风格的 image_url block（data URL 实测
    glm-5.3-flash 能读图），此前把 content 拍平成纯文本导致图片被静默
    丢弃、模型回答"没有看到图片"。
    """
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    parts: list[dict[str, Any]] = []
    if isinstance(content, list):
        for p in content:
            if not isinstance(p, dict):
                continue
            ptype = p.get("type")
            if ptype == "text":
                if p.get("text"):
                    parts.append({"type": "text", "text": p.get("text", "")})
            elif ptype == "image_url":
                iu = p.get("image_url")
                url = iu.get("url") if isinstance(iu, dict) else iu
                if url:
                    parts.append({"type": "image_url", "image_url": {"url": url}})
    return parts or [{"type": "text", "text": ""}]


def _extract_prompt(
    messages: list[dict[str, Any]],
    guard: bool = True,
    tools: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """OpenAI messages -> Trae messages（content 转成 block 数组）。

    纯对话请求（guard=True）注入 agent 协议压制指令：Trae 的 llm_utils_chat
    端点（solo_work_lite/chat_v3 等 function）服务端会给模型注入"你是带 shell
    工具的 coding agent"的预设，纯对话场景下模型会把 <Command>...</Command>
    之类工具调用语法当正文吐出来（下游 OpenAI 协议客户端不认识）。

    coding agent 请求（guard=False，由 _looks_like_agent_request 判定）不注入：
    下游 system prompt 自己教的工具格式模型会完美遵循，注入反而制造指令冲突。

    带 tools 的请求（prompt-based function calling shim）：
    - 工具教学指令追加到首条 system（没有则新建）
    - 历史 assistant.tool_calls 序列化成调用块文本（上游是纯 chat 模型，
      不认 tool_calls 字段，丢了模型就不知道自己之前调过什么）
    - role=tool 的结果消息转成 user 角色 + [tool_result] 前缀
    """
    out: list[dict[str, Any]] = []
    guard_merged = False
    tools_taught = False
    teaching = _build_tools_system(tools) if tools else ""

    def _content_parts(content: Any) -> list[dict[str, Any]]:
        return _content_blocks(content)


    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")

        if role == "system" and not guard_merged:
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text = "".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            else:
                text = ""
            merged = text
            if guard:
                merged = _AGENT_GUARD + "\n\n" + merged
            if teaching and not tools_taught:
                merged = merged + "\n\n" + teaching
                tools_taught = True
            out.append({"role": "system", "content": [{"type": "text", "text": merged}]})
            guard_merged = True
            continue

        parts = _content_parts(content)
        text = "".join(p["text"] for p in parts if p["type"] == "text")

        tool_calls = m.get("tool_calls") if isinstance(m, dict) else None
        if role == "assistant" and tool_calls:
            # 上游不认 tool_calls 字段：序列化成教学格式的调用块拼进正文
            block = _serialize_tool_calls(tool_calls)
            parts.append({"type": "text", "text": ("\n" + block) if text else block})
        elif role in ("tool", "function"):
            # 工具结果消息：转 user 角色 + [tool_result] 前缀（上游无此角色概念）
            name = m.get("name") or (
                (m.get("tool_call_id") and "") or "tool"
            )
            parts.insert(0, {"type": "text", "text": f"[tool_result | {name}]\n"})
            role = "user"

        out.append({"role": role, "content": parts})

    if guard and not guard_merged:
        out.insert(0, {
            "role": "system",
            "content": [{"type": "text", "text": _AGENT_GUARD}],
        })
    if teaching and not tools_taught:
        out.insert(0, {
            "role": "system",
            "content": [{"type": "text", "text": teaching}],
        })
    return out


def _build_chat_body(messages: list[dict[str, Any]], model: str, stream: bool) -> dict[str, Any]:
    session_id = str(uuid.uuid4())
    return {
        "messages": messages,
        "model": model,
        "config_name": model,  # Work 通道必须 config_name + model 成对（traework2api 实测）
        "function": "inline_chat",
        "stream": stream,
        "request_id": session_id,
        "session_id": session_id,
    }

