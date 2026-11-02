"""Finite evolution, immutable attempts, selection and held-out evaluation."""

import time
import re

from .store import digest
from .store import read, write, now, identity, inventory


def validate_splits(tasks):
    if set(tasks) != {"search", "selection", "final"}:
        raise ValueError("Expected search, selection, and final splits")
    seen_ids, seen_families, seen_templates, seen_cases = set(), set(), set(), set()
    for split, rows in tasks.items():
        if not rows or not any(r["case"]["problem"] == "healthy" for r in rows):
            raise ValueError(f"Split {split} needs a healthy control")
        families, templates = set(), set()
        for row in rows:
            if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", row["id"]):
                raise ValueError("Task id must be a single path component")
            if row["id"] in seen_ids or identity(row["case"]) in seen_cases:
                raise ValueError("Repeated task across splits")
            seen_ids.add(row["id"])
            seen_cases.add(identity(row["case"]))
            if row["case"]["problem"] != "healthy":
                families.add(row["family"])
                templates.add(row["template"])
        if families & seen_families or templates & seen_templates:
            raise ValueError("Fault family or template leakage across splits")
        seen_families |= families
        seen_templates |= templates


def verify_candidate(candidate):
    meta = read(candidate / "candidate.json")
    if digest(candidate / "candidate.py") != meta["sha256"]:
        raise ValueError("Candidate changed after staging")
    if (
        meta["status"] == "valid"
        and inventory(candidate / "project") != meta["project_hashes"]
    ):
        raise ValueError("Staged runtime changed after freezing")
    return meta


def evaluate_task(backend, candidate, split, row, deadline):
    """Resume one immutable task slot, retaining every failed attempt."""
    verify_candidate(candidate)
    slot = candidate / split / row["id"]
    attempts = sorted(slot.glob("attempt-*"))
    feedback = None
    for attempt in attempts:
        manifest = attempt / "artifacts.json"
        if manifest.exists() and inventory(attempt) != read(manifest):
            raise ValueError(f"Archived attempt changed: {attempt}")
        if (attempt / "feedback.json").exists():
            feedback = read(attempt / "feedback.json")
        else:
            recovery = slot / f"recovery-{attempt.name}.json"
            if not recovery.exists():
                cleaned = attempt / "cleanup.json"
                if (attempt / "artifacts.json").exists() and cleaned.exists():
                    cleanup = read(cleaned)
                    if not all(r["returncode"] == 0 for r in cleanup):
                        raise RuntimeError("Sealed attempt has unresolved cleanup")
                    write(recovery, {"already_cleaned": True, "at": now()})
                else:
                    backend.cleanup(attempt, log_path=recovery)
                    write(attempt / "artifacts.json", inventory(attempt))
    if not feedback or feedback["outcome"] not in {"success", "agent_failed"}:
        for number in range(len(attempts) + 1, backend.config["max_attempts"] + 1):
            if time.time() >= deadline:
                raise TimeoutError("Experiment wall-clock budget exhausted")
            feedback = backend.run(
                candidate / "project",
                row["case"],
                slot / f"attempt-{number:02d}",
                deadline,
            )
            if feedback["outcome"] in {"success", "agent_failed"}:
                break
    return feedback


def evaluate(backend, root, candidate, split, rows, deadline):
    meta = verify_candidate(candidate)
    if meta["status"] != "valid":
        summary = {
            "complete": True,
            "mean_rca_f1": 0.0,
            "candidate_failed": True,
            "reason": meta.get("reason"),
            "tasks": [],
        }
        write(candidate / split / "summary.json", summary)
        return summary
    results = []
    for row in rows:
        feedback = evaluate_task(backend, candidate, split, row, deadline)
        results.append({"task_id": row["id"], "feedback": feedback})
    scores = [r["feedback"]["score"] for r in results if r["feedback"] is not None]
    complete = (
        len(scores) == len(rows)
        and all(s is not None for s in scores)
        and all(r["feedback"].get("cleanup_ok", True) for r in results if r["feedback"])
    )
    summary = {
        "complete": complete,
        "mean_rca_f1": sum(scores) / len(scores) if complete else None,
        "tasks": results,
    }
    write(candidate / split / "summary.json", summary)
    return summary
