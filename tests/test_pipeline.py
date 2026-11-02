"""Exercise the complete official loop and recovery without external resources."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from nika_learning import meta_adapter
from nika_learning.experiment import evaluate, validate_splits
from nika_learning.store import read, write, inventory


def initialize(tmp_path):
    tasks = read(meta_adapter.ROOT / "configs/tasks.json")
    cfg = read(meta_adapter.ROOT / "configs/subscription.json")
    admission = tmp_path / "admission"
    for row in sum(tasks.values(), []):
        write(
            admission / (row["id"] + ".json"),
            {
                "admission": "pass",
                "case": row["case"],
                "nika_commit": cfg["nika_commit"],
            },
        )
    config = tmp_path / "config.json"
    cfg.update(
        rounds=2,
        total_timeout_sec=600,
        model="fixed",
        state_dir=str(tmp_path / "state"),
    )
    write(config, cfg)
    task_file = tmp_path / "tasks.json"
    write(task_file, tasks)
    root = tmp_path / "experiment"
    meta_adapter.create(root, config, task_file, admission)
    return root, tasks


class Backend:
    def __init__(self, *args):
        self.config = {"max_attempts": 2}
        self.executed, self.cleaned = [], []

    def cleanup(self, path, log_path=None):
        self.cleaned.append(path)
        write(log_path, [])

    def run(self, project, row, attempt, deadline):
        self.executed.append(project.parent.name)
        feedback = {"outcome": "success", "score": 0.5, "cleanup_ok": True}
        write(attempt / "feedback.json", feedback)
        write(attempt / "artifacts.json", inventory(attempt))
        return feedback


def test_two_rounds_candidate_failure_freeze_final_and_idempotent_resume(
    tmp_path, monkeypatch
):
    root, tasks = initialize(tmp_path)
    monkeypatch.setattr(meta_adapter, "NikaBackend", Backend)
    adapter = meta_adapter.Adapter(root, tmp_path)
    generated = []

    def propose(prompt, iteration, timeout=1200):
        generated.append(iteration)
        name = f"iter{iteration}_candidate"
        source = (
            "invalid python!"
            if iteration == 1
            else (meta_adapter.ROOT / "meta_candidates/baseline.py").read_text()
        )
        (adapter.workspace / "agents" / (name + ".py")).write_text(source)
        write(
            adapter.official.PENDING_EVAL,
            {
                "candidates": [
                    {"name": name, "import_path": f"agents.{name}:AgentHarness"}
                ]
            },
        )
        return True

    monkeypatch.setattr(adapter, "propose", propose)
    adapter.run("final")
    assert generated == [1, 2]
    assert "iter1_candidate" not in adapter.backend.executed
    assert "iter2_candidate" in adapter.backend.executed
    assert read(root / "frozen.json")["candidate"] == "baseline"
    assert read(root / "experiment.json")["status"] == "complete"
    assert set(read(root / "report.json")["final"]) == {"baseline"}
    assert not any(
        "selection" in str(p) or "final" in str(p)
        for p in (adapter.workspace / "history").rglob("feedback.json")
    )
    before = list(adapter.backend.executed)
    adapter.run("final")
    assert generated == [1, 2] and adapter.backend.executed == before


def test_resume_seals_interrupted_attempt_and_preserves_completed_work(tmp_path):
    root, tasks = initialize(tmp_path)
    candidate = root / "candidates/baseline"
    row = tasks["search"][0]
    first = candidate / "search" / row["id"] / "attempt-01"
    first.mkdir(parents=True)
    (first / "console.log").write_text("partial observation")
    backend = Backend()
    assert evaluate(backend, root, candidate, "search", [row], float("inf"))["complete"]
    assert backend.cleaned == [first]
    sealed = inventory(first)
    evaluate(backend, root, candidate, "search", [row], float("inf"))
    assert inventory(first) == sealed and len(backend.executed) == 1
    # Altering a scored attempt is detected before using its cached score.
    second = first.with_name("attempt-02")
    write(second / "feedback.json", {"outcome": "success", "score": 1})
    with pytest.raises(ValueError, match="Archived attempt changed"):
        evaluate(backend, root, candidate, "search", [row], float("inf"))


def test_fault_family_leak_and_path_traversal_are_rejected():
    tasks = read(meta_adapter.ROOT / "configs/tasks.json")
    tasks["final"][1]["family"] = tasks["search"][1]["family"]
    with pytest.raises(ValueError, match="leakage"):
        validate_splits(tasks)
    tasks["search"][0]["id"] = "../escape"
    with pytest.raises(ValueError, match="path component"):
        validate_splits(tasks)


def test_composed_primitive_calls_keep_observations_and_charge_each_call():
    import sys

    sys.path.insert(0, str(meta_adapter.ROOT / "meta_runtime"))
    from meta_agent.tools import ToolExecutor

    events, executed = [], []

    async def invoke(args):
        executed.append(args)
        return {"actual": args["host"]}

    tool = SimpleNamespace(
        name="probe",
        description="Observe",
        args_schema={"type": "object"},
        ainvoke=invoke,
    )
    log = SimpleNamespace(log=lambda name, value: events.append((name, value)))
    executor = ToolExecutor([tool], log, limit=2)

    async def composite():
        return [await executor.invoke("probe", {"host": name}) for name in ["a", "b"]]

    outputs = asyncio.run(composite())
    assert [json.loads(s)["observed"] for s in outputs] == [
        {"actual": "a"},
        {"actual": "b"},
    ]
    assert len([v for e, v in events if e == "tool_end"]) == 2
    with pytest.raises(RuntimeError, match="budget"):
        asyncio.run(executor.invoke("probe", {}))
    assert len(executed) == 2
