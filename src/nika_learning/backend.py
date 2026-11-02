"""Use NIKA's public CLI and original trial artifacts."""

import os
import signal
import subprocess
import time
from pathlib import Path

import httpx

from .store import read, write, inventory, now


def wait_process_group(group_id: int, timeout: float = 65):
    """Wait for live worker descendants after the CLI leader exits."""
    deadline = time.monotonic() + timeout
    while True:
        rows = subprocess.check_output(["ps", "-eo", "pgid=,stat="], text=True)
        live = any(
            int(parts[0]) == group_id and not parts[1].startswith("Z")
            for line in rows.splitlines()
            if len(parts := line.split()) == 2
        )
        if not live:
            return
        if time.monotonic() >= deadline:
            os.killpg(group_id, signal.SIGKILL)
            raise RuntimeError("Worker process group exceeded cleanup deadline")
        time.sleep(0.2)


def serving_identity(config):
    if config.get("model_api") == "responses":
        serving = {
            "provider": "codex_subscription",
            "model": config["model"],
            "model_api": "responses",
        }
    else:
        with httpx.Client(timeout=15) as client:
            models = client.get(config["base_url"] + "/models")
            models.raise_for_status()
            if config["model"] not in {m["id"] for m in models.json()["data"]}:
                raise RuntimeError(
                    "Fixed model is not ready; no trial resources were created"
                )
        with httpx.Client(timeout=15) as client:
            serving = client.get(config["base_url"].removesuffix("/v1") + "/status")
            serving.raise_for_status()
            serving = serving.json()
        model_backends = [
            b for b in serving["backends"] if b["model"] == config["model"]
        ]
        expected_jobs = {str(job) for job in config.get("serving_job_ids", [])}
        if not model_backends or any(
            str(b["job_id"]) not in expected_jobs or not b.get("ready", False)
            for b in model_backends
        ):
            raise RuntimeError(
                "Load balancer has an unexpected backend or no ready backend for the fixed model"
            )
    return serving


class NikaBackend:
    def __init__(self, checkout: Path, root: Path, config: dict):
        self.checkout, self.root, self.config = (
            checkout.resolve(),
            root.resolve(),
            config,
        )
        self.python = self.checkout / ".venv/bin/python"
        self.env = {**os.environ, "NIKA_AGENTS_FILE": str(self.root / "agents.yaml")}
        self.env.pop("NIKA_RUN_CONFIG", None)
        if config.get("state_dir"):
            self.env["NIKA_STATE_DIR"] = config["state_dir"]
        if config.get("docker_host"):
            self.env["DOCKER_HOST"] = config["docker_host"]
            self.env.pop("DOCKER_CONTEXT", None)
            self.env.pop("BUILDX_BUILDER", None)
            if config.get("state_dir"):
                self.env["BUILDX_CONFIG"] = str(Path(config["state_dir"]) / "buildx")
        self.env["PYTHONPATH"] = str(self.checkout / "src")
        actual = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != config["nika_commit"]:
            raise ValueError("NIKA commit differs from frozen configuration")
        if subprocess.check_output(
            ["git", "-C", str(checkout), "status", "--porcelain"], text=True
        ).strip():
            raise ValueError("NIKA checkout has tracked or untracked source changes")

    def command(self, *args, **kwargs):
        return subprocess.run(
            [str(self.python), "-m", "nika.cli.main", *args],
            cwd=self.checkout,
            env=self.env,
            **kwargs,
        )

    def cleanup(self, attempt: Path, log_path: Path | None = None):
        sessions = set()
        for path in attempt.rglob("run.json"):
            try:
                data = read(path)
                if data.get("session_id"):
                    sessions.add(data["session_id"])
            except (ValueError, OSError):
                continue
        results = []
        for sid in sorted(sessions):
            proc = self.command(
                "session",
                "close",
                "--session_id",
                sid,
                "--yes",
                capture_output=True,
                text=True,
                timeout=120,
            )
            results.append(
                {
                    "session_id": sid,
                    "returncode": proc.returncode,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                }
            )
        write(log_path or attempt / "cleanup.json", results)
        return results

    def run(self, project: Path, row: dict, attempt: Path, deadline: float):
        serving = serving_identity(self.config)
        attempt.mkdir(parents=True, exist_ok=False)
        write(attempt / "serving.json", serving)
        cfg = self.config
        run_config = {
            "agent": {
                "type": cfg.get("agent_type", "learning-pilot"),
                "provider": cfg.get("provider", "custom"),
                "model": cfg["model"],
                "base_url": cfg.get("base_url"),
                "max_steps": cfg["max_steps"],
                "max_tokens": cfg["max_tokens"],
                "timeout_sec": cfg["agent_timeout_sec"],
                "enable_skills": False,
                "submission": {"mode": "structured"},
            },
            "nika": {
                "result_dir": str(attempt / "nika"),
                "execution": {
                    "mode": cfg["execution_mode"],
                    **cfg.get("execution_options", {}),
                },
                "sandbox": {"backend": "sbx", "keep": False, "cpus": 2, "memory": "4g"},
            },
            "benchmark": {
                "batch_size": 1,
                "continue_on_error": True,
                "case_timeout_sec": cfg["case_timeout_sec"],
                "retry_passes": 0,
            },
        }
        run_config["agent"].update(
            fixed_model=True,
            tool_call_limits={
                "diagnosis": cfg["underlying_tool_calls"],
                "submission": 20,
            },
        )
        request_path = attempt / "rollout-request.json"
        write(
            request_path,
            {
                "schema_version": 1,
                "request_id": str(attempt.relative_to(self.root)),
                "task": row,
                "run_config": run_config,
                "agent_project": str(project),
            },
        )
        args = [
            str(self.python),
            "-m",
            "nika.cli.main",
            "rollout",
            "run",
            "--request",
            str(request_path),
            "--output-dir",
            str(attempt / "nika"),
        ]
        write(
            attempt / "identity.json",
            {
                "started": now(),
                "command": args,
                "model": cfg["model"],
                "nika_commit": cfg["nika_commit"],
                "task": row,
                "candidate_project": str(project),
                "resource_context": {
                    key: self.env.get(key)
                    for key in ["NIKA_STATE_DIR", "DOCKER_HOST", "BUILDX_CONFIG"]
                },
            },
        )
        started = time.monotonic()
        with (attempt / "console.log").open("w") as out:
            proc = subprocess.Popen(
                args,
                cwd=self.checkout,
                env=self.env,
                stdout=out,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            write(attempt / "process.json", {"pid": proc.pid, "started": now()})
            try:
                proc.wait(
                    timeout=max(
                        1, min(deadline - time.time(), cfg["case_timeout_sec"] + 180)
                    )
                )
            except BaseException:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=65)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                wait_process_group(proc.pid)
                self.cleanup(attempt)
                write(
                    attempt / "interrupted.json",
                    {"at": now(), "elapsed_sec": time.monotonic() - started},
                )
                write(attempt / "artifacts.json", inventory(attempt))
                raise
        cleanup = self.cleanup(attempt)
        feedback = self.collect(attempt)
        feedback["cleanup_ok"] = all(r["returncode"] == 0 for r in cleanup)
        feedback.update(
            returncode=proc.returncode, elapsed_sec=time.monotonic() - started
        )
        write(attempt / "feedback.json", feedback)
        write(attempt / "artifacts.json", inventory(attempt))
        return feedback

    @staticmethod
    def collect(attempt: Path):
        runs = list((attempt / "nika/native/trials").glob("*/run.json"))
        if len(runs) != 1:
            return {
                "outcome": "infra_failed",
                "score": None,
                "reason": "No unique native trial result",
            }
        native = read(runs[0])
        metrics_path = runs[0].with_name("eval_metrics.json")
        metrics = read(metrics_path) if metrics_path.exists() else {}
        outcome = native.get("outcome", "infra_failed")
        score = (
            metrics.get("rca_f1")
            if outcome == "success"
            else (0.0 if outcome == "agent_failed" else None)
        )
        required = [
            "run.json",
            "eval_metrics.json",
            "messages.jsonl",
            "model_calls.jsonl",
            "submission.json",
        ]
        missing = [name for name in required if not runs[0].with_name(name).exists()]
        if outcome == "success" and missing:
            score = None
        calls, usage, duration = 0, {}, 0.0
        for path in (attempt / "nika").rglob("model_calls.jsonl"):
            for line in path.read_text().splitlines():
                entry = __import__("json").loads(line)
                calls += int(entry.get("llm_request", False))
                duration += (entry.get("duration_ms", 0) or 0) / 1000
                for key, value in (entry.get("usage") or {}).items():
                    if isinstance(value, (int, float)):
                        usage[key] = usage.get(key, 0) + value
        return {
            "outcome": outcome,
            "score": score,
            "score_status": metrics.get("score_status") or native.get("score_status"),
            "validity_flags": metrics.get("validity_flags", []),
            "native_run": str(runs[0]),
            "native_metrics": str(metrics_path),
            "session_id": native.get("session_id"),
            "model_requests": calls,
            "model_elapsed_sec": duration,
            "usage": usage,
            "monetary_cost": None,
            "reason": native.get("error")
            or native.get("agent_error")
            or (
                f"Missing artifacts: {missing}"
                if outcome == "success" and missing
                else None
            ),
        }
