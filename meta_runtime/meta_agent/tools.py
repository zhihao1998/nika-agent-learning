"""Expose approved MCP primitives and record their actual observations."""

import json
import time
import uuid

from jsonschema import validate


class ToolExecutor:
    def __init__(self, tools, log, limit=60):
        self.raw = {tool.name: tool for tool in tools}
        if len(self.raw) != len(tools):
            raise ValueError("Ambiguous underlying tool names")
        self.log, self.limit, self.used = log, limit, 0
        self.specs = []
        for tool in tools:
            schema = tool.args_schema
            if not isinstance(schema, dict):
                schema = schema.model_json_schema()
            self.specs.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": schema,
                    },
                }
            )
        self.log.log("tool_inventory", {"tools": self.specs})

    async def invoke(self, name, arguments):
        tool = self.raw[name]
        schema = tool.args_schema
        if not isinstance(schema, dict):
            schema = schema.model_json_schema()
        validate(arguments, schema)
        if self.used >= self.limit:
            raise RuntimeError("Underlying tool call budget exhausted")
        self.used += 1
        call_id, started = uuid.uuid4().hex, time.monotonic()
        event = {"call_id": call_id, "tool": {"name": name}}
        self.log.log("tool_start", {**event, "input": arguments})
        try:
            result = await tool.ainvoke(arguments)
        except Exception as exc:
            self.log.log(
                "tool_error",
                {**event, "error": str(exc), "elapsed_sec": time.monotonic() - started},
            )
            raise
        self.log.log(
            "tool_end",
            {**event, "output": result, "elapsed_sec": time.monotonic() - started},
        )
        return json.dumps(
            {"observation_id": call_id, "tool": name, "observed": result},
            ensure_ascii=False,
            default=str,
        )
