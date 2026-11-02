import sys

import pytest

from nika_learning.meta_adapter import ROOT

sys.path.insert(0, str(ROOT / "meta_runtime"))
from meta_agent.transport import request, decode


def test_chat_preserves_tool_observation_pair_and_explicit_limits():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "probe", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "actual observation"},
    ]
    path, wire = request(
        "self-hosted",
        messages,
        [],
        {"model_api": "chat_completions", "temperature": 0.0, "max_tokens": 8192},
    )
    assert path == "/chat/completions"
    assert wire["messages"] == messages
    assert wire["temperature"] == 0.0 and wire["max_tokens"] == 8192
    assert "include" not in wire and "instructions" not in wire


def test_unknown_api_cannot_silently_use_subscription_transport():
    with pytest.raises(ValueError, match="Unsupported model API"):
        request("fixed", [], [], {"model_api": "unknown"})


def test_final_request_uses_existing_budget_and_cannot_request_more_tools():
    messages = [{"role": "user", "content": "Diagnose from real observations"}]
    tools = [{"type": "function", "function": {"name": "probe"}}]
    _, wire = request(
        "fixed",
        messages,
        tools,
        {"model_api": "chat_completions", "max_tokens": 8192, "temperature": 0},
        final=True,
    )
    assert wire["tools"] == [] and wire["tool_choice"] == "none"
    assert "final diagnosis report" in wire["messages"][-1]["content"]
    assert wire["messages"][:-1] == messages and len(messages) == 1
    assert tools[0]["function"]["name"] == "probe"


def test_chat_decoder_preserves_native_reasoning_and_counts_usage():
    import json

    message = {
        "role": "assistant",
        "content": None,
        "reasoning": "plan",
        "tool_calls": [],
    }
    result, usage = decode(
        "/chat/completions",
        json.dumps(
            {
                "choices": [{"finish_reason": "stop", "message": message}],
                "usage": {"prompt_tokens": 19, "completion_tokens": 7},
            }
        ),
    )
    assert result == message
    assert usage["input_tokens"] == 19 and usage["output_tokens"] == 7


def test_chat_proposer_uses_configured_service_without_subscription_credentials():
    from nika_learning.meta_proposer import upstream_for_config, proposer_request

    cfg = {
        "model_api": "chat_completions",
        "provider": "custom",
        "base_url": "http://serving/v1",
        "model": "fixed",
        "max_tokens": 8192,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    assert upstream_for_config(cfg) == ("http://serving/v1", "", {})
    endpoint, body = proposer_request(cfg, [{"role": "user", "content": "propose"}], [])
    assert endpoint == "/chat/completions"
    assert body["model"] == "fixed" and body["max_tokens"] == 8192
    assert body["chat_template_kwargs"]["enable_thinking"]
    assert "reasoning" not in body
