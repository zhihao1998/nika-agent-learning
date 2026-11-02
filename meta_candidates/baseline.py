"""Baseline diagnosis program for Meta-Harness code evolution."""

import json


class AgentHarness:
    async def run(self, task_description, ctx):
        messages = [
            {
                "role": "system",
                "content": "Diagnose the incident with live observations. Check healthy explanations. Confirm the root cause and affected resources. Return a concise report with observed evidence. Do not change network configuration.",
            },
            {"role": "user", "content": task_description},
        ]
        latest = ""
        for _ in range(ctx.max_steps):
            message = await ctx.complete(messages)
            messages.append(message)
            latest = message.get("content") or latest
            if not message.get("tool_calls"):
                return latest
            for call in message["tool_calls"]:
                arguments = json.loads(call["function"]["arguments"])
                output = await ctx.call_tool(call["function"]["name"], arguments)
                messages.append(
                    {"role": "tool", "tool_call_id": call["id"], "content": output}
                )
        return latest or "ERROR: No diagnosis report within the request budget"
