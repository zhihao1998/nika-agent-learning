import json

import pytest

from nika_learning.meta_adapter import check_source
from nika_learning.meta_proposer import operate, workspace_path


def test_execution_only_configuration_cannot_silently_skip_search(tmp_path):
    from nika_learning.meta_adapter import create

    config, tasks = tmp_path / "config.json", tmp_path / "tasks.json"
    config.write_text(json.dumps({"rounds": 0}))
    tasks.write_text("{}")
    root = tmp_path / "experiment"
    with pytest.raises(ValueError, match="at least one proposal round"):
        create(root, config, tasks, tmp_path / "admission")
    assert not root.exists()


def test_candidate_validation_does_not_execute_python(tmp_path):
    marker = tmp_path / "host-executed"
    source = f"from pathlib import Path\nPath({str(marker)!r}).touch()\nclass AgentHarness:\n async def run(self, task, ctx):\n  return 'report'\n"
    check_source(source)
    assert not marker.exists()


def test_proposer_filesystem_keeps_hidden_data_and_previous_candidates_immutable(
    tmp_path,
):
    workspace = tmp_path / "experiment/search_workspace"
    (workspace / "agents").mkdir(parents=True)
    (workspace / "agents/baseline.py").write_text("baseline")
    hidden = tmp_path / "experiment/final.json"
    hidden.write_text("secret")
    with pytest.raises(ValueError):
        workspace_path(workspace, "../final.json")
    (workspace / "escape").symlink_to(hidden)
    with pytest.raises(ValueError):
        workspace_path(workspace, "escape")
    with pytest.raises(ValueError):
        operate(
            workspace,
            1,
            "write_file",
            {"path": "agents/baseline.py", "content": "changed"},
        )
    operate(
        workspace, 1, "write_file", {"path": "agents/iter1_flow.py", "content": "new"}
    )
    with pytest.raises(ValueError):
        operate(
            workspace,
            1,
            "write_file",
            {"path": "agents/iter1_flow.py", "content": "changed"},
        )
    assert (
        json.loads(
            operate(workspace, 1, "read_file", {"path": "agents/iter1_flow.py"})
        )["content"]
        == "new"
    )


def test_history_exposes_search_interactions_but_not_private_execution_inputs(tmp_path):
    from nika_learning.meta_adapter import sync_history
    from nika_learning.store import write

    candidate = tmp_path / "candidates/baseline"
    candidate.mkdir(parents=True)
    (candidate / "candidate.py").write_text("source")
    for split in ["search", "selection", "final"]:
        directory = candidate / split / "task/attempt-01/nika/native/trials/trial"
        write(directory / "request.json", {"root_causes": ["private"]})
        write(directory / "run.json", {"inject": "private"})
        (directory / "messages.jsonl").write_text('{"observation":"actual"}\n')
    sync_history(tmp_path)
    history = tmp_path / "search_workspace/history/baseline"
    assert (history / "candidate.py").exists()
    assert len(list(history.rglob("messages.jsonl"))) == 1
    assert not list(history.rglob("request.json")) and not list(
        history.rglob("run.json")
    )
    assert not (history / "selection").exists() and not (history / "final").exists()


def test_proposer_requires_real_search_evidence_before_candidate_or_proposal(tmp_path):
    workspace = (tmp_path / "study/search_workspace").resolve()
    workspace.mkdir(parents=True)
    evidence = {}
    args = {"path": "agents/iter1_flow.py", "content": "source"}
    with pytest.raises(ValueError, match="search evidence"):
        operate(workspace, 1, "write_file", args, evidence=evidence)
    files = {
        "history/baseline/candidate.py": "source",
        "history/baseline/search/S01/attempt-01/feedback.json": '{"score":0}',
        "history/baseline/search/S01/attempt-01/messages.jsonl": '{"observation":"real"}',
    }
    for relative, body in files.items():
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body)
        # Empty pages do not count as reading an observation.
        operate(
            workspace,
            1,
            "read_file",
            {"path": relative, "offset": len(body)},
            evidence=evidence,
        )
    assert not evidence
    with pytest.raises(ValueError, match="search evidence"):
        operate(
            workspace,
            1,
            "write_file",
            {"path": "logs/study/pending_eval.json", "content": "{}"},
            evidence=evidence,
        )
    for relative in files:
        operate(workspace, 1, "read_file", {"path": relative}, evidence=evidence)
    operate(workspace, 1, "write_file", args, evidence=evidence)
    assert set(evidence) == {"code", "score", "trace"}
    assert (workspace / args["path"]).read_text() == "source"


def test_contract_failure_classification_keeps_infrastructure_errors_separate():
    from nika_learning.backend import trial_outcome

    candidate = {
        "outcome": "infra_failed",
        "agent_error": "Candidate contract error: import failed: missing module",
    }
    assert trial_outcome(candidate, {}) == "agent_failed"
    assert (
        trial_outcome(candidate, {"validity_flags": ["infra_error"]}) == "infra_failed"
    )
    assert (
        trial_outcome({"outcome": "infra_failed", "error": "sandbox unavailable"}, {})
        == "infra_failed"
    )


@pytest.mark.parametrize(
    "report", [None, {"diagnosis": "wrong type"}, "", "valid diagnosis"]
)
def test_runtime_enforces_report_contract_inside_candidate_boundary(
    monkeypatch, report
):
    import asyncio
    import importlib.util
    import sys
    from types import ModuleType, SimpleNamespace
    from nika_learning.meta_adapter import ROOT

    # Native SDK and MCP are boundary dependencies; this test creates no resources.
    stubs = {
        "agent.protocols": {"DIAGNOSIS": "diagnosis"},
        "agent.utils.loggers": {
            "MessageLogger": lambda **kw: SimpleNamespace(log=lambda *args: None)
        },
        "langchain_mcp_adapters.client": {
            "MultiServerMCPClient": lambda cfg: SimpleNamespace(get_tools=tools)
        },
        "meta_agent.submission": {"LearningAgent": type("LearningAgent", (), {})},
    }

    async def run(self, task, ctx):
        return report

    async def tools():
        return []

    stubs["harness"] = (
        {"AgentHarness": type("AgentHarness", (), {"run": run})}
        if report is not None
        else {}
    )
    for name, attrs in stubs.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location(
        "meta_agent.runtime_contract_test", ROOT / "meta_runtime/meta_agent/runtime.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    agent = module.MetaAgent()
    agent.model, agent.max_steps, agent.trace_dir = "fixed", 20, None
    agent.diagnosis_mcp_config = lambda: {}
    if report is None:
        with pytest.raises(
            RuntimeError, match="Candidate contract error: import failed"
        ):
            asyncio.run(agent.diagnose("task"))
    elif isinstance(report, str) and report:
        assert asyncio.run(agent.diagnose("task")) == report
    else:
        with pytest.raises(TypeError, match="Candidate contract error"):
            asyncio.run(agent.diagnose("task"))
