"""Use the configured provider API and retain its native tool-call protocol."""

import json
import os
from pathlib import Path

import httpx

from .responses import normalized, parse_sse, payload


def request(model, messages, tools, limits, *, final=False):
    if final:
        tools = []
        messages = [
            *messages,
            {
                "role": "user",
                "content": "The diagnosis request budget ends after this response. Return the final diagnosis report using existing observations. State uncertainty. Do not request tools.",
            },
        ]
    api = limits.get("model_api", "responses")
    if api == "responses":
        return "/responses", payload(model, messages, tools)
    if api != "chat_completions":
        raise ValueError("Unsupported model API: " + api)
    return "/chat/completions", {
        "model": model,
        "messages": messages,
        "tools": tools,
        "max_tokens": limits["max_tokens"],
        "temperature": limits["temperature"],
        "stream": False,
        **({"tool_choice": "none"} if final else {}),
        **(
            {"chat_template_kwargs": limits["chat_template_kwargs"]}
            if limits.get("chat_template_kwargs")
            else {}
        ),
    }


async def complete(model, messages, tools, *, final=False):
    limits = json.loads(Path(__file__).with_name("limits.json").read_text())
    endpoint, body = request(model, messages, tools, limits, final=final)
    async with httpx.AsyncClient(timeout=180) as client:
        async with client.stream(
            "POST",
            os.environ["OPENAI_BASE_URL"].rstrip("/") + endpoint,
            headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]},
            json=body,
        ) as response:
            response.raise_for_status()
            text = (await response.aread()).decode()
    return decode(endpoint, text)


def decode(endpoint, text):
    """Decode a native provider response without changing its tool messages."""
    if endpoint == "/responses":
        return normalized(parse_sse(text))
    data = json.loads(text)
    choice = data["choices"][0]
    if choice["finish_reason"] == "length":
        raise RuntimeError("Model completion exhausted the output-token budget")
    usage = data.get("usage", {})
    return choice["message"], {
        **usage,
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
    }
