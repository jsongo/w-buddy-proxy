"""豆包 agent 管线离线测试：payload 构造语义（不启动 CDP、不访问上游）。

覆盖 PR#28 的关键 wire 语义（App 抓包实测结论）：
- agent 管线 vs 经典管线的结构差异（agent_mode / model_config / aggregate_params）
- runtime_type 会话语义：新会话 2 / 续聊 1 + need_modify_conversation 取反
- reasoning_effort 传递与默认值
- 第三方模型（cis provider）字段

运行：PYTHONPATH=src python3 -m pytest test_doubao_agent_pipeline.py -v
"""
from __future__ import annotations

import json

import pytest

from buddy_proxy.doubao.cdp_client import CDPDoubaoClient
from buddy_proxy.doubao_provider import _DOUBAO_CHAT_MODELS


def _agent_payload(model_spec: dict, need_create: bool = True) -> dict:
    # _build_agent_payload 不依赖实例状态，用 None 作为 self 直接调用
    return CDPDoubaoClient._build_agent_payload(
        None, "你好", model_spec, need_create, "bot-1", "conv-1", 1700000000000, 1700000000,
    )


def test_agent_payload_core_fields():
    spec = {"item_key": "5", "extra": {"total_window_size": "256000"}, "provider": "",
            "reasoning_effort": 4}
    p = _agent_payload(spec, need_create=True)
    opt = p["option"]
    assert opt["agent_mode"] is 1 or opt["agent_mode"] == 1
    assert opt["model_config"] == {
        "model_item_key": "5",
        "model_extra_params": {"total_window_size": "256000"},
        "reasoning_effort": 4,
    }
    assert opt["aggregate_params"]["model_item_key"] == "5"
    assert opt["aggregate_params"]["agent_mode"] == "1"
    assert opt["aggregate_params"]["reasoning_effort"] == "4"
    assert opt["aggregate_params"]["provider_id"] == ""
    ext = p["ext"]
    assert ext["agent_mode"] == "1"
    # agent 管线里 use_deep_think 携带的是 model_item_key，不是 0/1/3 思考枚举
    assert ext["use_deep_think"] == "5"
    assert opt["need_deep_think"] == 5
    # 消息体走 block_type 10000 文本块
    assert p["messages"][0]["content_block"][0]["block_type"] == 10000
    assert p["messages"][0]["content_block"][0]["content"]["text_block"]["text"] == "你好"


def test_agent_payload_third_party_provider():
    """第三方模型（Gemini/GPT）带 provider_id=cis 与 memory_profile。"""
    spec = _DOUBAO_CHAT_MODELS["gemini-3.7-flash"]["agent"] | {"reasoning_effort": 3}
    p = _agent_payload(spec, need_create=True)
    assert p["option"]["aggregate_params"]["provider_id"] == "cis"
    assert p["option"]["model_config"]["model_extra_params"]["provider_id"] == "cis"
    # 大数 item_key 原样传递（gemini 的 model_item_key 是长数字串）
    assert p["option"]["model_config"]["model_item_key"] == "1946880770"
    assert p["ext"]["use_deep_think"] == "1946880770"


def test_agent_payload_runtime_type_new_vs_followup():
    """新会话 runtime_type=2 + 不改会话；续聊 runtime_type=1 + need_modify=True。"""
    gtp_new = json.loads(_agent_payload({"item_key": "9"}, need_create=True)["ext"]["general_task_param"])
    assert gtp_new["runtime_type"] == 2
    assert gtp_new["agent_task_param"]["runtime_type"] == 2
    assert gtp_new["need_modify_conversation"] is False

    gtp_follow = json.loads(_agent_payload({"item_key": "9"}, need_create=False)["ext"]["general_task_param"])
    assert gtp_follow["runtime_type"] == 1
    assert gtp_follow["agent_task_param"]["runtime_type"] == 1
    assert gtp_follow["need_modify_conversation"] is True

    # 续聊时 client_meta 带上会话 id；新会话 local_conversation_id 为空串
    p_follow = _agent_payload({"item_key": "9"}, need_create=False)
    assert p_follow["client_meta"]["conversation_id"] == "conv-1"
    assert p_follow["client_meta"]["local_conversation_id"] == ""


def test_model_table_shape():
    """模型表约束：agent 条目字段齐全；经典条目走 deep_think 枚举。"""
    classic = {k: v for k, v in _DOUBAO_CHAT_MODELS.items() if "agent" not in v}
    agent = {k: v for k, v in _DOUBAO_CHAT_MODELS.items() if "agent" in v}
    assert {"doubao", "doubao-pro", "doubao-think", "doubao-expert"} <= set(classic)
    assert {"doubao-auto", "doubao-2.1-turbo", "doubao-2.1-pro",
            "orange-5.0", "gemini-3.7-flash", "gpt-5.6-sol"} <= set(agent)
    for mid, spec in agent.items():
        assert "item_key" in spec["agent"], mid
        assert "desc" in spec, mid
    for mid, spec in classic.items():
        assert spec["deep_think"] in (0, 1, 3), mid


def test_classic_payload_has_no_agent_fields():
    """经典管线不携带 agent 字段（服务端忽略模型字段，固定默认豆包）。"""
    p = CDPDoubaoClient._build_classic_payload(
        None, "你好", 0, True, "bot-1", "conv-1", 1700000000000, 1700000000,
    )
    assert "agent_mode" not in p["option"]
    assert "model_config" not in p["option"]
    assert "aggregate_params" not in p["option"]
