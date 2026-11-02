"""NIKA hooks around the unmodified official Meta-Harness Terminal-Bench loop."""

import argparse
import ast
import fcntl
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

from .backend import NikaBackend, wait_process_group
from .store import digest
from .experiment import evaluate, validate_splits, verify_candidate
from .store import read, write, now, identity, inventory

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "third_party/meta_harness"


def upstream_module():
    provenance = read(UPSTREAM / "UPSTREAM.json")
    for relative, expected in provenance["files"].items():
        if digest(UPSTREAM / relative) != expected:
            raise ValueError("Vendored official source differs: " + relative)
    import sys

    sys.path.insert(0, str(UPSTREAM / "terminal_bench_2"))
    spec = importlib.util.spec_from_file_location(
        "official_meta_harness", UPSTREAM / "terminal_bench_2/meta_harness.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_source(source):
    tree = ast.parse(source)
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AgentHarness"
    )
    if not any(
        isinstance(n, ast.AsyncFunctionDef) and n.name == "run" for n in cls.body
    ):
        raise ValueError("AgentHarness must define async run(task_description, ctx)")
    compile(tree, "candidate.py", "exec")


def stage(root, name, source, *, config=None):
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ValueError("Invalid candidate name")
    candidate = root / "candidates" / name
    if candidate.exists():
        verify_candidate(candidate)
        if (candidate / "candidate.py").read_text() != source:
            raise ValueError("Candidate source changed after staging")
        return candidate
    candidate.mkdir(parents=True)
    (candidate / "candidate.py").write_text(source)
    try:
        check_source(source)
        project = candidate / "project"
        shutil.copytree(
            ROOT / "meta_runtime", project, ignore=shutil.ignore_patterns("__pycache__")
        )
        (project / "harness.py").write_text(source)
        if config is None:
            config = (
                read(root / "experiment.json")["config"]
                if (root / "experiment.json").exists()
                else {}
            )
        write(
            project / "meta_agent/limits.json",
            {
                "model_api": config.get("model_api", "responses"),
                "max_tokens": config.get("max_tokens", 8192),
                "temperature": config.get("temperature", 0.0),
                "chat_template_kwargs": config.get("chat_template_kwargs", {}),
            },
        )
        write(
            candidate / "candidate.json",
            {
                "status": "valid",
                "sha256": digest(candidate / "candidate.py"),
                "project_hashes": inventory(project),
                "created": now(),
            },
        )
    except Exception as exc:
        write(
            candidate / "candidate.json",
            {
                "status": "candidate_failed",
                "sha256": digest(candidate / "candidate.py"),
                "reason": str(exc),
                "created": now(),
            },
        )
    return candidate


def create(root, config_path, tasks_path, admission_dir):
    config, tasks = read(config_path), read(tasks_path)
    if not isinstance(config.get("rounds"), int) or config["rounds"] < 1:
        raise ValueError(
            "Meta-Harness requires at least one proposal round; staged configs are execution checks"
        )
    if config.get("model_api") == "chat_completions" and not config.get(
        "serving_job_ids"
    ):
        raise ValueError(
            "Set serving_job_ids to the fixed load-balancer job before initialization"
        )
    if config.get("model_api") not in {"responses", "chat_completions"}:
        raise ValueError("Set model_api to responses or chat_completions")
    if not Path(config["state_dir"]).is_absolute():
        raise ValueError("state_dir must be absolute")
    for key in [
        "max_steps",
        "max_tokens",
        "optimizer_requests",
        "max_attempts",
        "agent_timeout_sec",
        "case_timeout_sec",
        "total_timeout_sec",
        "underlying_tool_calls",
    ]:
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError("Expected a positive integer: " + key)
    if config["underlying_tool_calls"] > 60:
        raise ValueError("underlying_tool_calls exceeds the fixed runtime cap of 60")
    validate_splits(tasks)
    records = {}
    for row in sum(tasks.values(), []):
        record = read(admission_dir / (row["id"] + ".json"))
        if (
            record["admission"] != "pass"
            or record["case"] != row["case"]
            or record.get("nika_commit") != config["nika_commit"]
        ):
            raise ValueError("Task admission failed or stale: " + row["id"])
        records[row["id"]] = record
    root.mkdir(parents=True, exist_ok=False)
    for name, record in records.items():
        write(root / "admission" / (name + ".json"), record)
    write(
        root / "experiment.json",
        {
            "created": now(),
            "started_epoch": time.time(),
            "config": config,
            "tasks": tasks,
            "contract_hash": identity([config, tasks]),
            "runtime_hashes": inventory(ROOT / "meta_runtime"),
            "controller_hashes": inventory(ROOT / "src/nika_learning"),
            "upstream_hashes": inventory(UPSTREAM),
            "admission_hashes": inventory(root / "admission"),
            "status": "created",
            "method": "official-meta-harness-terminal-bench-loop-nika-adapter",
        },
    )
    workspace = root / "search_workspace"
    (workspace / "agents").mkdir(parents=True)
    shutil.copyfile(
        ROOT / "meta_candidates/baseline.py", workspace / "agents/baseline.py"
    )
    shutil.copyfile(ROOT / "domain_spec.md", workspace / "domain_spec.md")
    shutil.copyfile(
        ROOT / "meta_runtime/meta_agent/runtime.py", workspace / "runtime-interface.py"
    )
    stage(root, "baseline", (workspace / "agents/baseline.py").read_text())


def sync_history(root):
    history = root / "search_workspace/history"
    for candidate in (root / "candidates").iterdir():
        if not candidate.is_dir():
            continue
        for path in candidate.rglob("*"):
            rel = path.relative_to(candidate)
            if path.is_file() and (
                rel == Path("candidate.py")
                or (
                    rel.parts[0] == "search"
                    and path.name
                    in {
                        "summary.json",
                        "feedback.json",
                        "messages.jsonl",
                        "model_calls.jsonl",
                        "tool_calls.jsonl",
                        "submission.json",
                    }
                )
            ):
                target = history / candidate.name / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)

    for path in (root / "rounds").rglob("*"):
        if path.is_file() and path.name in {
            "messages.json",
            "exchanges.json",
            "model_calls.jsonl",
            "pending_eval.json",
            "failure.json",
            "interrupted.json",
        }:
            target = (
                root
                / "search_workspace/logs/proposer"
                / path.relative_to(root / "rounds")
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)


class Adapter:
    def __init__(self, root, checkout):
        self.root = root
        self.meta = read(root / "experiment.json")
        self.config = self.meta["config"]
        self.backend = NikaBackend(checkout, root, self.config)
        self.deadline = self.meta["started_epoch"] + self.config["total_timeout_sec"]
        self.official = upstream_module()
        self.workspace = root / "search_workspace"

    def candidate(self, name, import_path):
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError("Invalid candidate name")
        if import_path != f"agents.{name}:AgentHarness":
            raise ValueError("Candidate name must match its module")
        source = (self.workspace / "agents" / (name + ".py")).read_text()
        return stage(self.root, name, source)

    def evaluate_search(self, import_path, job_name, n_trials=1, n_concurrent=1):
        if n_trials != 1 or n_concurrent != 1:
            raise ValueError("This pilot fixes one trial and one concurrent worker")
        name = import_path.split(":")[0].split(".")[-1]
        candidate = self.candidate(name, import_path)
        summary = evaluate(
            self.backend,
            self.root,
            candidate,
            "search",
            self.meta["tasks"]["search"],
            self.deadline,
        )
        if not summary["complete"]:
            raise RuntimeError("Unresolved infrastructure failure: " + name)
        job = self.official.JOBS_DIR / job_name
        write(job / "nika-summary.json", summary)
        write(job / "config.json", {"model": self.config["model"], "n_attempts": 1})
        write(
            job / "result.json",
            {"stats": {"n_errors": 0}, "source": "native-nika-feedback"},
        )
        sync_history(self.root)
        return job, True

    def validate(self, name, import_path):
        try:
            candidate = self.candidate(name, import_path)
            return verify_candidate(candidate)["status"] == "valid"
        except Exception as exc:
            write(
                self.official.LOGS_DIR / f"validation-{name}.json",
                {"error": str(exc), "candidate": name},
            )
            return False

    def smoke(self, name, import_path, timeout=1800):
        # The first search task doubles as a real sandbox smoke check.
        candidate = self.candidate(name, import_path)
        summary = evaluate(
            self.backend,
            self.root,
            candidate,
            "search",
            self.meta["tasks"]["search"][:1],
            self.deadline,
        )
        sync_history(self.root)
        if not summary["complete"]:
            raise RuntimeError("Smoke has an unresolved infrastructure failure")
        return True  # Counted agent failures remain in the full evaluation.

    def propose(self, prompt, iteration, timeout=1200):
        if time.time() >= self.deadline:
            raise TimeoutError("Experiment wall-clock budget exhausted")
        directory = self.root / "rounds" / f"iter{iteration}"
        saved = directory / "pending_eval.json"
        if saved.exists():
            write(self.official.PENDING_EVAL, read(saved))
            return True
        sync_history(self.root)
        pending = self.official.PENDING_EVAL.relative_to(self.workspace)
        prompt = (
            f"Run iteration {iteration} of the official Meta-Harness loop adapted to NIKA.\n"
            "Read domain_spec.md and runtime-interface.py. Read prior candidate source, scores, successful and failed traces.\n"
            "The search space is executable Python diagnosis code. Change one mechanism supported by trace evidence.\n"
            "You may implement context management, diagnostic control flow, tool wrappers, composite tools, parsing or prompts.\n"
            "Do not hardcode task names, node names, fault locations or answers from history. Do not change the fixed runtime.\n"
            f"Write one new agents/iter{iteration}_name.py. It must define class AgentHarness with async run(task_description, ctx).\n"
            "Use ctx.complete(messages, tools=None) for model requests. Use ctx.call_tool(name, arguments) for approved primitive tools.\n"
            "ctx.tools contains OpenAI function schemas. You may transform exposed schemas and implement dispatch in your candidate.\n"
            "Return a plain-text diagnosis report. Preserve typed tool calls and tool outputs in model context.\n"
            "Check Python syntax with check_python. Do not execute benchmarks.\n"
            f"Write {pending} as JSON: "
            + json.dumps(
                {
                    "iteration": iteration,
                    "candidates": [
                        {
                            "name": f"iter{iteration}_name",
                            "import_path": f"agents.iter{iteration}_name:AgentHarness",
                            "hypothesis": "one falsifiable mechanism",
                            "changes": "code changes",
                            "expected_efficiency": "expected request/token impact",
                        }
                    ],
                }
            )
            + "\n"
            f"The evaluated model is {self.config['model']}. Search covers {len(self.meta['tasks']['search'])} cases.\n"
            f"Official frontier and summary are under logs/{self.root.name}/. Full NIKA history is under history/.\n"
        )
        directory.mkdir(parents=True, exist_ok=True)
        previous_attempts = len(list(directory.glob("attempt-*")))
        if previous_attempts >= self.config["max_attempts"]:
            raise RuntimeError("Proposer retries exhausted for this round")
        attempt = directory / f"attempt-{previous_attempts + 1:02d}"
        attempt.mkdir()
        (attempt / "prompt.txt").write_text(prompt)
        env = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                [str(ROOT / "src"), str(self.backend.checkout / "src")]
            ),
        }
        command = [
            str(self.backend.python),
            "-m",
            "nika_learning.meta_proposer",
            str(self.root),
            str(self.workspace),
            str(iteration),
            str(attempt / "prompt.txt"),
            str(attempt),
        ]
        with (attempt / "console.log").open("w") as log:
            result = subprocess.Popen(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                result.wait(timeout=min(timeout, max(1, self.deadline - time.time())))
            except BaseException:
                os.killpg(result.pid, signal.SIGTERM)
                try:
                    result.wait(timeout=65)
                except subprocess.TimeoutExpired:
                    os.killpg(result.pid, signal.SIGKILL)
                    result.wait()
                wait_process_group(result.pid)
                write(attempt / "interrupted.json", {"at": now()})
                write(attempt / "artifacts.json", inventory(attempt))
                raise
        write(attempt / "artifacts.json", inventory(attempt))
        if result.returncode or not self.official.PENDING_EVAL.exists():
            write(
                directory / "failure.json",
                {"returncode": result.returncode, "at": now()},
            )
            raise RuntimeError(
                "Proposer failed; resume retries this round within max_attempts"
            )
        proposal = read(self.official.PENDING_EVAL)
        if len(proposal.get("candidates", [])) != 1:
            write(
                directory / "failure.json", {"error": "Expected exactly one candidate"}
            )
            raise ValueError("Proposer must submit exactly one candidate")
        write(saved, proposal)
        return True

    def wire(self):
        m = self.official
        m.EVOLVE_DIR = self.workspace
        m.AGENTS_DIR = self.workspace / "agents"
        m.BASELINES = [("baseline", "agents.baseline:AgentHarness")]
        m.BASELINE_AGENT_NAME = "baseline"
        m.MODEL = self.config["model"]
        m.N_EVAL_TASKS = len(self.meta["tasks"]["search"])
        m.harbor_run = self.evaluate_search
        m.validate_candidate = self.validate
        m.smoke_test = self.smoke
        m.propose_claude = self.propose
        m.render_task_prompt = (
            lambda iteration, n_trials: ""
        )  # Domain prompt is built by the proposer hook.
        m.parse_job_results = (
            lambda job_dir, expected_trials=None: {
                row["task_id"]: [row["feedback"]["score"]]
                for row in read(job_dir / "nika-summary.json")["tasks"]
            }
            if (job_dir / "nika-summary.json").exists()
            else {}
        )
        # Native RCA F1 is continuous; do not turn every partial score into a full pass.
        m.compute_pass_rates = lambda rewards: (
            {key: sum(values) / len(values) for key, values in rewards.items()},
            sum(sum(values) for values in rewards.values())
            / sum(len(values) for values in rewards.values())
            if rewards
            else 0.0,
        )

        def trial_metrics(job_dir):
            result = {}
            for row in read(job_dir / "nika-summary.json")["tasks"]:
                feedback = row["feedback"]
                usage = feedback.get("usage") or {}
                result[row["task_id"]] = [
                    {
                        "n_input_tokens": usage.get("input_tokens"),
                        "n_output_tokens": usage.get("output_tokens"),
                        "n_cache_tokens": usage.get("cache_read_input_tokens"),
                        "n_api_calls": feedback.get("model_requests"),
                        "n_turns": feedback.get("model_requests"),
                        "cost_usd": None,
                        "reward": feedback["score"],
                    }
                ]
            return result

        m.parse_trial_metrics = trial_metrics
        return m

    def run(self, until):
        m = self.wire()
        m.LOGS_DIR = self.workspace / "logs" / self.root.name
        m.JOBS_DIR = self.workspace / "jobs" / self.root.name
        m.EVOLUTION_SUMMARY = m.LOGS_DIR / "evolution_summary.jsonl"
        done = m.count_iterations()
        if done < self.config["rounds"]:
            m.run_evolve(
                SimpleNamespace(
                    run_name=self.root.name,
                    fresh=False,
                    skip_baseline=False,
                    iterations=self.config["rounds"] - done,
                    trials=1,
                    concurrent=1,
                    propose_timeout=1200,
                    skip_smoke=False,
                    full_eval=False,
                )
            )
        sync_history(self.root)
        if m.count_iterations() != self.config["rounds"]:
            raise RuntimeError("Search did not complete all requested rounds")
        write(
            self.root / "search-complete.json",
            {"at": now(), "rounds": m.count_iterations()},
        )
        if until == "search":
            return
        candidates = ["baseline"]
        for line in m.EVOLUTION_SUMMARY.read_text().splitlines():
            name = json.loads(line)["agent"]
            if name not in candidates and (self.root / "candidates" / name).exists():
                candidates.append(name)
        selection = []
        for name in candidates:
            summary = evaluate(
                self.backend,
                self.root,
                self.root / "candidates" / name,
                "selection",
                self.meta["tasks"]["selection"],
                self.deadline,
            )
            if not summary["complete"]:
                raise RuntimeError("Selection has unresolved infrastructure failures")
            selection.append((name, summary["mean_rca_f1"]))
        winner = max(selection, key=lambda item: item[1])[0]
        frozen = {
            "candidate": winner,
            "sha256": digest(self.root / "candidates" / winner / "candidate.py"),
            "contract_hash": self.meta["contract_hash"],
            "selection": selection,
        }
        if (self.root / "frozen.json").exists() and read(
            self.root / "frozen.json"
        ) != json.loads(json.dumps(frozen)):
            raise ValueError("Frozen candidate changed")
        write(self.root / "frozen.json", frozen)
        if until == "selection":
            return
        final = {
            name: evaluate(
                self.backend,
                self.root,
                self.root / "candidates" / name,
                "final",
                self.meta["tasks"]["final"],
                self.deadline,
            )
            for name in dict.fromkeys(["baseline", winner])
        }
        if not all(summary["complete"] for summary in final.values()):
            raise RuntimeError(
                "Final evaluation has unresolved infrastructure failures"
            )
        write(
            self.root / "report.json",
            {"completed": now(), "frozen": frozen, "final": final},
        )
        self.meta["status"] = "complete"
        write(self.root / "experiment.json", self.meta)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("experiment", type=Path)
    init.add_argument("--config", type=Path, required=True)
    init.add_argument("--tasks", type=Path, required=True)
    init.add_argument("--admission", type=Path, required=True)
    for verb in ["run", "resume"]:
        run = sub.add_parser(verb)
        run.add_argument("experiment", type=Path)
        run.add_argument("--checkout", type=Path, default=ROOT / ".deps/nika")
        run.add_argument(
            "--until", choices=["search", "selection", "final"], default="final"
        )
    args = p.parse_args()
    root = args.experiment.resolve()
    if args.command == "init":
        create(root, args.config, args.tasks, args.admission)
        return

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    with (root / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        meta = read(root / "experiment.json")
        for key, path in [
            ("runtime_hashes", ROOT / "meta_runtime"),
            ("controller_hashes", ROOT / "src/nika_learning"),
            ("upstream_hashes", UPSTREAM),
            ("admission_hashes", root / "admission"),
        ]:
            if meta[key] != inventory(path):
                raise ValueError("Frozen experiment files changed: " + key)
        if meta["contract_hash"] != identity([meta["config"], meta["tasks"]]):
            raise ValueError("Experiment contract changed")
        Adapter(root, args.checkout.resolve()).run(args.until)


if __name__ == "__main__":
    main()
