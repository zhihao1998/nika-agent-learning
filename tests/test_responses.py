"""Responses requests preserve tool and reasoning state without credentials."""

import sys

import pytest

from nika_learning.meta_adapter import ROOT

sys.path.insert(0, str(ROOT / "meta_runtime"))
from meta_agent.responses import payload, parse_sse, normalized


def test_preserves_reasoning_tool_call_and_observation_pair():
    output = [
        {"type": "reasoning", "id": "r1", "encrypted_content": "opaque"},
        {"type": "function_call", "call_id": "c1", "name": "probe", "arguments": "{}"},
    ]
    message, _ = normalized({"output": output, "usage": {"input_tokens": 10}})
    wire = payload(
        "gpt-6-luna",
        [
            {"role": "system", "content": "Diagnose"},
            message,
            {"role": "tool", "tool_call_id": "c1", "content": "actual observation"},
        ],
        [],
    )
    assert wire["input"][:2] == output
    assert wire["input"][2] == {
        "type": "function_call_output",
        "call_id": "c1",
        "output": "actual observation",
    }
    assert wire["store"] is False and wire["stream"] is True
    assert wire["reasoning"]["effort"] == "low"


def test_refuses_incomplete_sse_and_reports_failed_response():
    with pytest.raises(RuntimeError, match="before response.completed"):
        parse_sse('data: {"type":"response.created"}\n')
    with pytest.raises(RuntimeError, match="failed"):
        parse_sse('data: {"type":"response.failed"}\n')
