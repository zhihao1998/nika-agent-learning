"""Verify native identities, fixed budgets, immutable attempts and image pairing."""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


from nika_learning.store import read, write, inventory, identity
from nika_learning.meta_adapter import ROOT, UPSTREAM
from nika_learning.store import digest
from nika_learning.experiment import verify_candidate

sys.path.insert(0, str(ROOT / "meta_runtime"))
from meta_agent.responses import parse_sse

p = argparse.ArgumentParser()
p.add_argument("experiment", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
root = a.experiment.resolve()
meta = read(root / "experiment.json")
cfg = meta["config"]
errors = []
for key, path in [
    ("runtime_hashes", ROOT / "meta_runtime"),
    ("controller_hashes", ROOT / "src/nika_learning"),
    ("upstream_hashes", UPSTREAM),
    ("admission_hashes", root / "admission"),
]:
    if inventory(path) != meta[key]:
        errors.append("Frozen source mismatch: " + key)
if identity([cfg, meta["tasks"]]) != meta["contract_hash"]:
    errors.append("Experiment contract changed")
for candidate in (root / "candidates").iterdir():
    if candidate.is_dir():
        try:
            verify_candidate(candidate)
        except Exception as exc:
            errors.append(f"Candidate archive changed: {candidate.name}: {exc}")
for path in root.glob("candidates/*/*/summary.json"):
    summary = read(path)
    if not summary["complete"] or summary.get("candidate_failed"):
        continue
    split = path.parent.name
    expected_ids = {task["id"] for task in meta["tasks"][split]}
    if {task["task_id"] for task in summary["tasks"]} != expected_ids:
        errors.append(
            "Summary does not contain exactly its fixed task split: " + str(path)
        )
    scores = []
    for task in summary["tasks"]:
        attempts = sorted(
            (path.parent / task["task_id"]).glob("attempt-*/feedback.json")
        )
        if not attempts or read(attempts[-1]) != task["feedback"]:
            errors.append(
                "Summary feedback differs from archived attempt: " + str(path)
            )
        scores.append(task["feedback"]["score"])
    if not scores or any(score is None for score in scores):
        errors.append("Complete summary contains unresolved reward: " + str(path))
    elif abs(sum(scores) / len(scores) - summary["mean_rca_f1"]) > 1e-12:
        errors.append("Summary mean differs from per-task scores: " + str(path))
if (root / "frozen.json").exists():
    frozen = read(root / "frozen.json")
    winner = frozen["candidate"]
    if frozen["sha256"] != digest(root / "candidates" / winner / "candidate.py"):
        errors.append("Frozen candidate source changed")
    if frozen["contract_hash"] != meta["contract_hash"]:
        errors.append("Frozen experiment contract changed")
    selection = frozen["selection"]
    if max(selection, key=lambda item: item[1])[0] != winner:
        errors.append("Selection did not apply stable maximum rule")
    for name, score in selection:
        summary = read(root / "candidates" / name / "selection/summary.json")
        if not summary["complete"] or summary["mean_rca_f1"] != score:
            errors.append("Frozen selection score differs from summary: " + name)
    evaluated_final = {path.parent.name for path in root.glob("candidates/*/final")}
    if evaluated_final - {"baseline", winner}:
        errors.append("Final data evaluated an unselected candidate")
settings = set()
tasks = defaultdict(set)
images = defaultdict(set)
outcomes = Counter()
usage = Counter()
requests = Counter()
tool_calls = Counter()
manifests = 0
feedbacks = 0
for manifest in list(root.glob("candidates/*/*/*/attempt-*/artifacts.json")) + list(
    root.glob("rounds/*/attempt-*/artifacts.json")
):
    manifests += 1
    if inventory(manifest.parent) != read(manifest):
        errors.append("Archived attempt changed: " + str(manifest))
for path in root.glob("candidates/*/*/*/attempt-*/nika/request.json"):
    request = read(path)
    run_config = request["run_config"]
    run_config["nika"].pop("result_dir", None)
    settings.add(identity(run_config))
    relative = path.relative_to(root).parts
    tasks[(relative[2], relative[3])].add(identity(request["task"]))
for path in root.glob("candidates/*/*/*/attempt-*/feedback.json"):
    feedbacks += 1
    feedback = read(path)
    outcomes[feedback["outcome"]] += 1
    runs = list((path.parent / "nika/native/trials").glob("*/run.json"))
    if len(runs) != 1:
        errors.append("No unique native trial: " + str(path))
        continue
    native = read(runs[0])
    metrics_path = runs[0].with_name("eval_metrics.json")
    metrics = read(metrics_path) if metrics_path.exists() else {}
    expected = (
        metrics.get("rca_f1")
        if feedback["outcome"] == "success"
        else (0.0 if feedback["outcome"] == "agent_failed" else None)
    )
    if expected != feedback["score"] or native["session_id"] != feedback["session_id"]:
        errors.append("Score identity mismatch: " + str(path))
    if not feedback.get("cleanup_ok"):
        errors.append("Cleanup failed: " + str(path))
    relative = path.relative_to(root).parts
    fc = native.get("firecracker", {})
    images[(relative[2], relative[3])].add((fc.get("image"), fc.get("case")))
    if cfg["execution_mode"] == "firecracker" and not fc.get("image"):
        errors.append("Missing Firecracker identity: " + str(path))
for path in list(
    root.glob("candidates/*/*/*/attempt-*/nika/native/trials/*/model_calls.jsonl")
) + list(root.glob("rounds/*/attempt-*/model_calls.jsonl")):
    phase_counts = Counter()
    kind = "optimizer" if "rounds" in path.relative_to(root).parts else "agent"
    for line in path.read_text().splitlines():
        event = json.loads(line)
        if not event.get("llm_request"):
            if (
                event.get("method") == "POST"
                and event["status"] < 400
                and not event.get("path", "").endswith("/messages/count_tokens")
            ):
                errors.append("Unmetered successful model POST: " + str(path))
            continue
        requests[kind] += 1
        refused = event["status"] == 400 and "budget (max_steps)" in str(
            event.get("response")
        )
        charged = event["status"] != 429 and event["status"] < 500 and not refused
        phase_counts[event["phase"]] += int(charged)
        if event["status"] == 200:
            body = event["response"]
            chat = cfg.get("model_api") == "chat_completions"
            if isinstance(body, str):
                body = json.loads(body) if chat else parse_sse(body)
            if (
                event["request"].get("model") != cfg["model"]
                or body.get("model", cfg["model"]) != cfg["model"]
            ):
                errors.append("Served a different model: " + str(path))
            expected_effort = (
                cfg.get("optimizer_reasoning_effort", "low")
                if kind == "optimizer"
                else cfg.get("model_reasoning_effort", "low")
            )
            inference_matches = (
                event["request"].get("temperature") == cfg["temperature"]
                and event["request"].get("max_tokens") == cfg["max_tokens"]
                and event["request"].get("chat_template_kwargs", {})
                == cfg.get("chat_template_kwargs", {})
                if chat
                else event["request"].get("reasoning", {}).get("effort")
                == expected_effort
            )
            if not inference_matches:
                errors.append(
                    "Inference effort differs from configuration: " + str(path)
                )
            raw = body.get("usage", {})
            if chat:
                raw = {
                    **raw,
                    "input_tokens": raw.get("prompt_tokens", 0),
                    "output_tokens": raw.get("completion_tokens", 0),
                }
            for key in ["input_tokens", "output_tokens"]:
                if event.get("usage", {}).get(key, 0) != raw.get(key, 0):
                    errors.append("Raw usage mismatch: " + str(path))
        for key, value in (event.get("usage") or {}).items():
            if isinstance(value, (int, float)):
                usage[key] += value
    cap = cfg["optimizer_requests"] if kind == "optimizer" else cfg["max_steps"]
    if any(value > cap for value in phase_counts.values()):
        errors.append("Model budget exceeded: " + str(path))
for path in root.glob(
    "candidates/*/*/*/attempt-*/nika/native/trials/*/tool_calls.jsonl"
):
    counts = Counter()
    for line in path.read_text().splitlines():
        event = json.loads(line)
        if event.get("denied"):
            tool_calls["denied"] += 1
            continue
        counts[event["phase"]] += 1
        tool_calls[event["phase"]] += 1
        if (
            event.get("call_id") is None
            or event.get("duration_ms") is None
            or event.get("raw_output") is None
        ):
            errors.append("Missing primitive provenance: " + str(path))
    if counts["diagnosis"] > cfg["underlying_tool_calls"] or counts["submission"] > 20:
        errors.append("Primitive budget exceeded: " + str(path))
if len(settings) != 1:
    errors.append("Trial configurations differ beyond result paths")
if any(len(value) != 1 for value in tasks.values()):
    errors.append("Task changed across candidates or attempts")
if any(len(value) != 1 for value in images.values()):
    errors.append("Restored image pairing changed")
record = {
    "experiment": str(root),
    "status": meta["status"],
    "feedbacks": feedbacks,
    "artifact_manifests": manifests,
    "configuration_variants": len(settings),
    "task_slots": len(tasks),
    "outcomes": dict(outcomes),
    "model_requests": dict(requests),
    "usage": dict(usage),
    "primitive_calls": dict(tool_calls),
    "paired_images": {
        f"{key[0]}/{key[1]}": list(value) for key, value in images.items()
    },
    "errors": errors,
}
write(a.output, record)
print(json.dumps(record, indent=2))
if errors:
    raise SystemExit(1)
