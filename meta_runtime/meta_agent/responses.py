"""Responses transport for the Codex-compatible subscription endpoint."""

import json


def payload(model, messages, tools):
    instructions, items = [], []
    for message in messages:
        role = message.get("role")
        if role == "system":
            instructions.append(message["content"])
        elif message.get("_responses_output") is not None:
            items.extend(message["_responses_output"])
        elif role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": message["content"],
                }
            )
        elif role == "assistant":
            if message.get("content"):
                items.append({"role": "assistant", "content": message["content"]})
            for call in message.get("tool_calls", []):
                items.append(
                    {"type": "function_call", "call_id": call["id"], **call["function"]}
                )
        else:
            items.append({"role": role, "content": message["content"]})
    return {
        "model": model,
        "instructions": "\n".join(instructions),
        "input": items,
        "tools": [
            {"type": "function", **t["function"], "strict": False} for t in tools
        ],
        "store": False,
        "stream": True,
        "reasoning": {"effort": "low"},
        "include": ["reasoning.encrypted_content"],
    }


def parse_sse(text):
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        data = json.loads(line[6:])
        if data.get("type") == "response.completed":
            return data["response"]
        if data.get("type") in {"response.failed", "error"}:
            raise RuntimeError("Responses request failed: " + json.dumps(data))
    raise RuntimeError("Stream ended before response.completed")


def normalized(response):
    calls, texts = [], []
    for item in response["output"]:
        if item["type"] == "function_call":
            calls.append(
                {
                    "id": item["call_id"],
                    "type": "function",
                    "function": {"name": item["name"], "arguments": item["arguments"]},
                }
            )
        if item["type"] == "message":
            texts.extend(
                c["text"] for c in item["content"] if c["type"] == "output_text"
            )
    message = {
        "role": "assistant",
        "content": "\n".join(texts),
        "_responses_output": response["output"],
    }
    if calls:
        message["tool_calls"] = calls
    return message, response.get("usage", {})
