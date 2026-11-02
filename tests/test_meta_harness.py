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
