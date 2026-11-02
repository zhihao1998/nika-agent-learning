"""Index every score by immutable candidate, attempt and native session identity."""

import argparse
import csv
from pathlib import Path

from nika_learning.store import read

p = argparse.ArgumentParser()
p.add_argument("experiment", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
root = a.experiment.resolve()
meta = read(root / "experiment.json")
columns = [
    "experiment",
    "contract_hash",
    "candidate",
    "candidate_sha256",
    "split",
    "task_id",
    "attempt",
    "outcome",
    "rca_f1",
    "score_status",
    "session_id",
    "model_requests",
    "input_tokens",
    "output_tokens",
    "elapsed_sec",
    "cleanup_ok",
    "model_calls",
    "tool_calls",
    "submission",
    "metrics",
    "native_run",
    "reason",
]
a.output.parent.mkdir(parents=True, exist_ok=True)
with a.output.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    for feedback_path in sorted(root.glob("candidates/*/*/*/attempt-*/feedback.json")):
        feedback = read(feedback_path)
        parts = feedback_path.relative_to(root).parts
        candidate, split, task, attempt = parts[1:5]
        candidate_meta = read(root / "candidates" / candidate / "candidate.json")
        runs = list((feedback_path.parent / "nika/native/trials").glob("*/run.json"))
        native = runs[0].parent if len(runs) == 1 else None

        def reference(name):
            return (
                str((native / name).relative_to(root))
                if native is not None and (native / name).exists()
                else ""
            )

        usage = feedback.get("usage") or {}
        writer.writerow(
            {
                "experiment": root.name,
                "contract_hash": meta["contract_hash"],
                "candidate": candidate,
                "candidate_sha256": candidate_meta["sha256"],
                "split": split,
                "task_id": task,
                "attempt": attempt,
                "outcome": feedback["outcome"],
                "rca_f1": feedback["score"],
                "score_status": feedback.get("score_status"),
                "session_id": feedback.get("session_id"),
                "model_requests": feedback.get("model_requests"),
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "elapsed_sec": feedback.get("elapsed_sec"),
                "cleanup_ok": feedback.get("cleanup_ok"),
                "model_calls": reference("model_calls.jsonl"),
                "tool_calls": reference("tool_calls.jsonl"),
                "submission": reference("submission.json"),
                "metrics": reference("eval_metrics.json"),
                "native_run": reference("run.json"),
                "reason": feedback.get("reason"),
            }
        )
