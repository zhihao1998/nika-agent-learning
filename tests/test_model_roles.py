import pytest

from nika_learning.meta_proposer import proposer_config, proposer_request
from nika_learning.gateway import inference_headers


def test_proposer_does_not_inherit_executor_transport_or_thinking():
    config = {
        "model": "small",
        "model_api": "responses",
        "gateway": {"cluster": "rocs"},
        "chat_template_kwargs": {"enable_thinking": True},
        "max_tokens": 4096,
        "optimizer_requests": 24,
        "temperature": 0.0,
        "proposer": {
            "model": "large",
            "model_api": "chat_completions",
            "provider": "custom",
            "base_url": "http://localhost/v1",
            "serving_job_ids": [123],
        },
    }
    resolved = proposer_config(config)
    _, body = proposer_request(resolved, [], [])
    assert body["model"] == "large" and "gateway" not in resolved
    assert body.get("chat_template_kwargs", {}) == {}
    assert config["model"] == "small" and config["gateway"]["cluster"] == "rocs"
    with pytest.raises(ValueError, match="serving identity"):
        proposer_config({"proposer": {"model": "large"}})


def test_orix_requires_approval_and_private_credentials(tmp_path):
    token = tmp_path / "token"
    token.write_text("test-secret")
    token.chmod(0o600)
    cfg = {
        "model": "orix/large",
        "base_url": "http://127.0.0.1:18766/v1",
        "gateway": {"cluster": "orix", "token_file": str(token), "session": "test"},
    }
    with pytest.raises(ValueError, match="approval"):
        inference_headers(cfg)
    cfg["gateway"]["external_data_approved"] = True
    assert inference_headers(cfg)["X-Compute-Project"] == "nika"
    token.chmod(0o644)
    with pytest.raises(ValueError, match="600"):
        inference_headers(cfg)


def test_context_budget_preserves_tool_pairs_and_original_evidence():
    import json
    from nika_learning.meta_proposer import bounded_messages

    prefix = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "task"},
    ]
    old = [
        {"role": "assistant", "tool_calls": [{"id": "old"}]},
        {"role": "tool", "tool_call_id": "old", "content": "x" * 2000},
    ]
    recent = [
        {"role": "assistant", "tool_calls": [{"id": "new"}]},
        {"role": "tool", "tool_call_id": "new", "content": "actual trace"},
    ]
    messages = prefix + old + recent
    result = bounded_messages(messages, max_chars=600)
    assert result == prefix + recent
    assert messages == prefix + old + recent
    assert len(json.dumps(result)) <= 600
    with pytest.raises(ValueError, match="exchange"):
        bounded_messages(prefix + old, max_chars=600)


def test_proposer_continues_truncation_without_executing_partial_tool_calls():
    import json
    from nika_learning.meta_proposer import decode_proposal
    from meta_agent.transport import decode

    body = json.dumps(
        {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {
                        "role": "assistant",
                        "content": "partial code",
                        "tool_calls": [{"id": "partial"}],
                    },
                }
            ]
        }
    )
    message, truncated = decode_proposal("/chat/completions", body)
    assert truncated and message["content"] == "partial code"
    assert "tool_calls" not in message
    with pytest.raises(RuntimeError, match="output-token budget"):
        decode("/chat/completions", body)


@pytest.mark.parametrize("request_budget", [1, 2])
def test_proposer_finishes_when_tool_submits_candidate(
    tmp_path, monkeypatch, request_budget
):
    import json
    import sys
    from contextlib import nullcontext
    from types import ModuleType

    import httpx
    from nika_learning import backend, meta_proposer

    model_proxy = ModuleType("nika.agents.sandbox.model_proxy")
    model_proxy.model_proxy = lambda *args, **kwargs: nullcontext(1234)
    model_proxy.new_proxy_token = lambda: "test-token"
    model_calls = ModuleType("nika.agents.sandbox.model_calls")
    model_calls.ModelCallLog = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, model_proxy.__name__, model_proxy)
    monkeypatch.setitem(sys.modules, model_calls.__name__, model_calls)

    root = tmp_path / "study"
    workspace = root / "search_workspace"
    workspace.mkdir(parents=True)
    config = {
        "model": "large",
        "model_api": "chat_completions",
        "provider": "custom",
        "base_url": "http://unused/v1",
        "optimizer_requests": request_budget,
        "max_tokens": 4096,
        "temperature": 0.0,
    }
    (root / "experiment.json").write_text(json.dumps({"config": config}))
    reads = {
        "history/baseline/candidate.py": "source",
        "history/baseline/search/summary.json": "{}",
        "history/baseline/search/messages.jsonl": '{"observation":"actual"}',
    }
    for relative, content in reads.items():
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    source = (
        "class AgentHarness:\n async def run(self, task, ctx):\n  return 'diagnosis'\n"
    )
    pending = {
        "iteration": 1,
        "candidates": [
            {"name": "iter1_flow", "import_path": "agents.iter1_flow:AgentHarness"}
        ],
    }
    actions = [("read_file", {"path": p}) for p in reads] + [
        ("write_file", {"path": "agents/iter1_flow.py", "content": source}),
        ("check_python", {"path": "agents/iter1_flow.py"}),
        (
            "write_file",
            {"path": "logs/study/pending_eval.json", "content": json.dumps(pending)},
        ),
    ]
    calls = []

    def respond(request):
        calls.append(request)
        assert len(calls) == 1, "A submitted proposal needs no final assistant turn"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": str(i),
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": json.dumps(args),
                                    },
                                }
                                for i, (name, args) in enumerate(actions)
                            ],
                        },
                    }
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(meta_proposer.httpx, "Client", lambda **kwargs: client)
    monkeypatch.setattr(backend, "serving_identity", lambda cfg: {})
    meta_proposer.generate(root, workspace, 1, "propose", root / "attempt")
    assert (
        json.loads((workspace / "logs/study/pending_eval.json").read_text()) == pending
    )
    assert (workspace / "agents/iter1_flow.py").read_text() == source
    assert len(calls) == 1
