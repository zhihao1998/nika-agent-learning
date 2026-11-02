"""Run NIKA's existing admission audit before learning sees task feedback."""

import json
import sys
import subprocess
from pathlib import Path

from experiment.audit.live import audit_case
from nika.config import REPO_ROOT
from nika_learning.experiment import validate_splits


def main():
    source, destination = map(Path, sys.argv[1:])
    destination.mkdir(parents=True, exist_ok=True)
    tasks = json.loads(source.read_text())
    validate_splits(tasks)
    commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    results = {}
    for split, rows in tasks.items():
        for row in rows:
            path = destination / (row["id"] + ".json")
            if path.exists():
                record = json.loads(path.read_text())
                if record["case"] != row["case"] or record.get("nika_commit") != commit:
                    raise ValueError(f"Existing admission has a different case: {path}")
            else:
                print("Auditing", split, row["id"], flush=True)
                try:
                    result = audit_case(row["case"])
                    record = {
                        "task_id": row["id"],
                        "case": row["case"],
                        "admission": result.admission(),
                        "audit": result.model_dump(mode="json"),
                    }
                except Exception as exc:
                    record = {
                        "task_id": row["id"],
                        "case": row["case"],
                        "admission": "error",
                        "error": str(exc),
                    }
                record["nika_commit"] = commit
                path.write_text(json.dumps(record, indent=2) + "\n")
            results[row["id"]] = record["admission"]
            print(row["id"], record["admission"], flush=True)
    (destination / "report.json").write_text(
        json.dumps(
            {"tasks": results, "complete": all(v == "pass" for v in results.values())},
            indent=2,
        )
        + "\n"
    )
    return 0 if all(v == "pass" for v in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
