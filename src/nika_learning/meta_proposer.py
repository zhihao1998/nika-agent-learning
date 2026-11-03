"""Metered provider transport and filesystem tools for the official proposer."""

import json
import signal
import sys
from contextlib import ExitStack
from pathlib import Path

import httpx

from .store import read, write

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "meta_runtime"))
from meta_agent.transport import request, decode  # noqa: E402


def upstream_for_config(cfg):
    if cfg.get("model_api", "responses") == "chat_completions":
        if cfg.get("provider") != "custom" or not cfg.get("base_url"):
            raise ValueError("Chat proposer requires a custom provider and base_url")
        # The existing self-hosted endpoint is unauthenticated. Keep subscription
        # credentials out of custom endpoints and experiment configuration.
        return cfg["base_url"], "", {}
    if cfg.get("model_api", "responses") != "responses":
        raise ValueError("Unsupported proposer model API")
    from nika.agents.sandbox.model_proxy import _subscription_upstream

    return _subscription_upstream(
        anthropic=False, provider="openai", env_file=Path(".env")
    )


def proposer_request(cfg, messages, tools):
    endpoint, body = request(cfg["model"], messages, tools, cfg)
    if endpoint == "/responses":
        body["reasoning"] = {"effort": cfg.get("optimizer_reasoning_effort", "low")}
    return endpoint, body


def workspace_path(workspace, relative):
    path = (workspace / relative).resolve()
    if not path.is_relative_to(workspace.resolve()) or not path.is_file():
        raise ValueError("File is outside the search workspace")
    return path


def evidence_kind(relative):
    """Identify search evidence, excluding proposer prose and hidden splits."""
    parts = Path(relative).parts
    if len(parts) < 3 or parts[0] != "history":
        return None
    if parts[2:] == ("candidate.py",):
        return "code"
    if len(parts) > 3 and parts[2] == "search":
        if parts[-1] in {"feedback.json", "summary.json"}:
            return "score"
        if parts[-1] in {"messages.jsonl", "model_calls.jsonl", "tool_calls.jsonl"}:
            return "trace"
    return None


def operate(workspace, iteration, name, args, *, evidence=None):
    if name == "list_files":
        return "\n".join(
            str(p.relative_to(workspace))
            for p in sorted(workspace.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts
        )
    if name == "read_file":
        p = workspace_path(workspace, args["path"])
        offset = max(0, int(args.get("offset", 0)))
        body = p.read_text()
        page = body[offset : offset + 24000]
        if evidence is not None and page.strip():
            kind = evidence_kind(p.relative_to(workspace.resolve()))
            if kind:
                evidence.setdefault(kind, []).append(
                    {
                        "path": str(p.relative_to(workspace.resolve())),
                        "offset": offset,
                        "chars": len(page),
                    }
                )
        return json.dumps(
            {
                "path": args["path"],
                "offset": offset,
                "total_chars": len(body),
                "content": page,
            }
        )
    if name == "write_file":
        relative = Path(args["path"])
        path = (workspace / relative).resolve()
        pending = workspace / "logs" / workspace.parent.name / "pending_eval.json"
        candidate = (
            len(relative.parts) == 2
            and relative.parts[0] == "agents"
            and relative.name.startswith(f"iter{iteration}_")
            and relative.suffix == ".py"
        )
        if not path.is_relative_to(workspace.resolve()) or not (
            candidate or path == pending
        ):
            raise ValueError(
                "Write a new agents/iterN_name.py or the requested pending_eval.json only"
            )
        if path.exists() and path != pending:
            raise ValueError(
                "A submitted candidate is immutable; choose another new filename"
            )
        if evidence is not None:
            missing = {"code", "score", "trace"} - evidence.keys()
            if missing:
                raise ValueError(
                    "Read historical search evidence before writing a candidate: "
                    + ", ".join(sorted(missing))
                )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args["content"])
        return "Written " + str(relative)
    if name == "check_python":
        p = workspace_path(workspace, args["path"])
        from .meta_adapter import check_source

        check_source(p.read_text())
        return "Syntax and interface checks passed. Runtime validation occurs in the NIKA sandbox."
    raise ValueError("Unknown proposer tool")


def generate(root, workspace, iteration, prompt, log_dir):
    from nika.agents.sandbox.model_calls import ModelCallLog
    from nika.agents.sandbox.model_proxy import (
        model_proxy,
        new_proxy_token,
    )

    cfg = read(root / "experiment.json")["config"]
    messages = [
        {
            "role": "system",
            "content": "You are the coding proposer for Meta-Harness. Follow the supplied domain specification. Read prior code and complete search traces before proposing one executable Python harness. Use filesystem tools to implement the candidate and pending_eval.json. Do not fabricate evidence. Do not access selection or final data.",
        },
        {"role": "user", "content": prompt},
    ]
    tools = []
    for name, description, properties, required in [
        ("list_files", "List files in the search workspace.", {}, []),
        (
            "read_file",
            "Read a complete file in pages. Use offset to continue reading.",
            {"path": {"type": "string"}, "offset": {"type": "integer"}},
            ["path"],
        ),
        (
            "write_file",
            "Write a new candidate source or the requested pending evaluation file.",
            {"path": {"type": "string"}, "content": {"type": "string"}},
            ["path", "content"],
        ),
        (
            "check_python",
            "Check candidate Python syntax and its asynchronous interface without executing code.",
            {"path": {"type": "string"}},
            ["path"],
        ),
    ]:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                        "additionalProperties": False,
                    },
                },
            }
        )
    upstream, credential, headers = upstream_for_config(cfg)
    token = new_proxy_token()
    exchanges = []
    evidence = {}
    log_dir.mkdir(parents=True, exist_ok=True)
    from .backend import serving_identity

    write(log_dir / "serving.json", serving_identity(cfg))
    try:
        with ExitStack() as stack:
            port = stack.enter_context(
                model_proxy(
                    upstream,
                    bind_host="127.0.0.1",
                    token=token,
                    api_key=credential,
                    extra_headers=headers,
                    call_log=ModelCallLog(
                        log_dir / "model_calls.jsonl",
                        max_requests=cfg["optimizer_requests"],
                        expected_model=cfg["model"],
                    ),
                )
            )
            client = stack.enter_context(httpx.Client(timeout=240))
            for _ in range(cfg["optimizer_requests"]):
                endpoint, body = proposer_request(cfg, messages, tools)
                response = client.post(
                    f"http://127.0.0.1:{port}" + endpoint,
                    json=body,
                    headers={"Authorization": "Bearer " + token},
                )
                exchanges.append(
                    {
                        "request": body,
                        "status": response.status_code,
                        "response": response.text,
                    }
                )
                write(log_dir / "exchanges.json", exchanges)
                response.raise_for_status()
                message, _ = decode(endpoint, response.text)
                messages.append(message)
                calls = message.get("tool_calls", [])
                if not calls:
                    pending = workspace / "logs" / root.name / "pending_eval.json"
                    if pending.exists():
                        return
                    messages.append(
                        {
                            "role": "user",
                            "content": "Implement one candidate and write pending_eval.json before finishing.",
                        }
                    )
                    continue
                for call in calls:
                    try:
                        args = json.loads(call["function"]["arguments"])
                        result = operate(
                            workspace,
                            iteration,
                            call["function"]["name"],
                            args,
                            evidence=evidence,
                        )
                    except Exception as exc:
                        result = json.dumps({"error": str(exc)})
                    messages.append(
                        {"role": "tool", "tool_call_id": call["id"], "content": result}
                    )
            raise RuntimeError("Proposer request budget exhausted")
    finally:
        write(log_dir / "evidence-reads.json", evidence)
        write(log_dir / "messages.json", messages)


if __name__ == "__main__":

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    root, workspace, iteration, prompt_file, log_dir = sys.argv[1:]
    generate(
        Path(root),
        Path(workspace),
        int(iteration),
        Path(prompt_file).read_text(),
        Path(log_dir),
    )
