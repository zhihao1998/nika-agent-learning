"""Run the frozen diagnosis policy inside the standard sandbox."""

import json
from langchain_mcp_adapters.client import MultiServerMCPClient
from agent.base import DiagnosisAgent
from agent.utils.loggers import MessageLogger

from .tools import ToolExecutor
from .transport import complete


class LearningAgent(DiagnosisAgent):
    async def submit(self, diagnosis_report, context):
        from agent.utils.mcp_client import session_mcp_config
        from agent.utils.submission_context import submission_single_prompt

        log = MessageLogger(phase="submission", session_dir=self.trace_dir)
        tools = await MultiServerMCPClient(session_mcp_config("submission")).get_tools()
        executor = ToolExecutor(tools, log, limit=20)
        messages = [
            {
                "role": "system",
                "content": "Submit the frozen diagnosis using the submit tool. Do not query the network.",
            },
            {
                "role": "user",
                "content": submission_single_prompt(diagnosis_report, context),
            },
        ]
        for _ in range(self.max_steps):
            message, usage = await complete(self.model, messages, executor.specs)
            log.log(
                "llm_end",
                {
                    "text": message.get("content", ""),
                    "model": self.model,
                    "usage_metadata": usage,
                },
            )
            messages.append(message)
            for call in message.get("tool_calls", []):
                result = await executor.invoke(
                    call["function"]["name"],
                    json.loads(call["function"]["arguments"]),
                )
                messages.append(
                    {"role": "tool", "tool_call_id": call["id"], "content": result}
                )
                if (
                    call["function"]["name"] == "submit"
                    and "submission rejected" not in result.lower()
                ):
                    return result
        raise RuntimeError("Submission request budget exhausted")
