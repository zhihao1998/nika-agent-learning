"""Run candidate Python only inside NIKA's normal sandbox."""

import asyncio
import json
from agent.protocols import DIAGNOSIS
from agent.utils.loggers import MessageLogger
from langchain_mcp_adapters.client import MultiServerMCPClient

from .submission import LearningAgent
from .transport import complete
from .tools import ToolExecutor


class HarnessContext:
    """Convenient model and primitive MCP services; the host enforces limits."""

    def __init__(self, agent, executor, log):
        self.model = agent.model
        self.max_steps = agent.max_steps
        self.tools = executor.specs
        self._executor, self._log = executor, log
        self._requests = 0

    async def complete(self, messages, tools=None):
        """Return a typed assistant message, retaining actual provider responses."""
        self._requests += 1
        if self._requests > self.max_steps:
            raise RuntimeError("Diagnosis request budget exhausted")
        message, usage = await complete(
            self.model,
            messages,
            self.tools if tools is None else tools,
            final=self._requests == self.max_steps,
        )
        self._log.log(
            "llm_end",
            {
                "text": message.get("content") or "",
                "model": self.model,
                "usage_metadata": usage,
            },
        )
        return message

    async def call_tool(self, name, arguments):
        """Execute an approved primitive and return its observed output."""
        try:
            return await self._executor.invoke(name, arguments)
        except Exception as exc:
            return json.dumps({"error": str(exc), "observation": False})


class MetaAgent(LearningAgent):
    async def diagnose(self, task_description):
        from harness import AgentHarness

        log = MessageLogger(phase=DIAGNOSIS, session_dir=self.trace_dir)
        client = MultiServerMCPClient(self.diagnosis_mcp_config())
        executor = ToolExecutor(await client.get_tools(), log, limit=60)
        async with asyncio.timeout(600):
            return await AgentHarness().run(
                task_description, HarnessContext(self, executor, log)
            )

    async def submit(self, diagnosis_report, context):
        # Candidate code is not imported in this phase.
        return await super().submit(diagnosis_report, context)
