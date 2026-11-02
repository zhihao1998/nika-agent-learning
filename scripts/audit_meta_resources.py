"""Inspect experiment-owned sessions, sandbox workspaces and Firecracker copies."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import yaml

from nika_learning.store import now, read, write

parser = argparse.ArgumentParser()
parser.add_argument("output", type=Path)
parser.add_argument("--run", action="append", type=Path, required=True)
parser.add_argument(
    "--profile", action="append", nargs=2, metavar=("NIKA", "CONFIG"), required=True
)
args = parser.parse_args()
names, workspaces, sessions, trials, owners = set(), set(), set(), set(), set()
for root in args.run:
    for path in root.rglob("run.json"):
        if "search_workspace" in path.parts:
            continue
        data = read(path)
        if data.get("session_id"):
            sessions.add(data["session_id"])
        if data.get("trial_id"):
            trials.add(data["trial_id"])
        if path.parent.name == "native":
            owners.add(str(path.parent.resolve()))
    for path in root.rglob("nika.jsonl"):
        if "search_workspace" in path.parts:
            continue
        for line in path.read_text().splitlines():
            event = json.loads(line)
            data = event.get("data") or {}
            if data.get("session_id"):
                sessions.add(data["session_id"])
            if data.get("sandbox_name"):
                names.add(data["sandbox_name"])
            if event.get("event") == "sandbox_start":
                workspace = (data.get("env") or {}).get("AGENT_WORKSPACE")
                if workspace:
                    workspaces.add(workspace)
for name in names:
    # Setup can fail before sandbox_start. Its native sandbox_cleanup event
    # still identifies the opaque workspace that must have been removed.
    if name.startswith("agent-") and "/" not in name:
        workspaces.add(
            str(
                Path(f"/tmp/agent-workspaces-{os.getuid()}")
                / name.removeprefix("agent-")
            )
        )
listed = json.loads(subprocess.check_output(["sbx", "list", "--json"], text=True))
remaining = [row for row in listed["sandboxes"] if row["name"] in names]
profiles = []
for index, (checkout_arg, config_arg) in enumerate(args.profile):
    checkout, config = Path(checkout_arg).resolve(), read(Path(config_arg))
    # Inspection uses the normal native settings loader, with this profile's
    # exact state and cache root, rather than the user's default registry.
    native_config = {
        "nika": {
            "execution": {
                "mode": config["execution_mode"],
                **config.get("execution_options", {}),
            }
        }
    }
    config_path = args.output.parent / f"resource-audit-profile-{index}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(native_config))
    env = {
        **os.environ,
        "PYTHONPATH": str(checkout / "src"),
        "NIKA_STATE_DIR": config["state_dir"],
        "NIKA_RUN_CONFIG": str(config_path.resolve()),
    }
    if config.get("docker_host"):
        env["DOCKER_HOST"] = config["docker_host"]
        env.pop("DOCKER_CONTEXT", None)
    code = """import json
from nika.run_config import get_run_config
from nika.firecracker.trial import trial_rows
from nika.workflows.session.list import list_sessions
print(json.dumps({'running_sessions':list_sessions(running_only=True), 'trial_copies':trial_rows() if get_run_config().nika.execution.mode == 'firecracker' else []}, default=str))
"""
    data = json.loads(
        subprocess.check_output(
            [str(checkout / ".venv/bin/python"), "-c", code], env=env, text=True
        )
    )
    data["unrelated_running_session_count"] = sum(
        r.get("session_id") not in sessions for r in data["running_sessions"]
    )
    data["running_sessions"] = [
        r for r in data["running_sessions"] if r.get("session_id") in sessions
    ]
    data["unrelated_trial_copy_count"] = sum(
        r.get("trial") not in trials | sessions and r.get("owner") not in owners
        for r in data["trial_copies"]
    )
    data["trial_copies"] = [
        r
        for r in data["trial_copies"]
        if r.get("trial") in trials | sessions or r.get("owner") in owners
    ]
    data["checkout"] = str(checkout)
    data["config"] = config_arg
    data["cache_root"] = (
        config.get("execution_options", {}).get("firecracker", {}).get("root")
    )
    profiles.append(data)
proof = {
    "checked_at": now(),
    "runs": [str(root.resolve()) for root in args.run],
    "recorded_owned_sessions": sorted(sessions),
    "recorded_owned_sandbox_names": sorted(names),
    "remaining_owned_sandboxes": remaining,
    "remaining_owned_workspaces": [
        path for path in sorted(workspaces) if Path(path).exists()
    ],
    "profiles": profiles,
    "unrelated_sandbox_count": len(listed["sandboxes"]) - len(remaining),
}
proof["clean"] = not (
    remaining
    or proof["remaining_owned_workspaces"]
    or any(row["running_sessions"] or row["trial_copies"] for row in profiles)
)
write(args.output, proof)
print(json.dumps(proof, indent=2))
if not proof["clean"]:
    raise SystemExit(1)
