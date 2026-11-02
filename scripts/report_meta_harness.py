"""Report native candidate scores separately from the best-so-far search curve."""

import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("experiment", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
root = a.experiment.resolve()
meta = json.loads((root / "experiment.json").read_text())
summary_path = root / "search_workspace/logs" / root.name / "evolution_summary.jsonl"
rows = (
    [json.loads(line) for line in summary_path.read_text().splitlines()]
    if summary_path.exists()
    else []
)
names = list(dict.fromkeys(["baseline"] + [r["agent"] for r in rows]))
records, curve = [], []
best = None
for name in names:
    iteration = next((r["iteration"] for r in rows if r["agent"] == name), 0)
    candidate = root / "candidates" / name
    record = {"iteration": iteration, "candidate": name, "splits": {}}
    for split in ["search", "selection", "final"]:
        path = candidate / split / "summary.json"
        if path.exists():
            record["splits"][split] = json.loads(path.read_text())
            cost = {
                "model_requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "model_elapsed_sec": 0.0,
                "executed_primitive_calls": 0,
                "attempts": 0,
                "attempt_elapsed_sec": 0.0,
            }
            # Include interrupted and retried attempts, rather than charging
            # only the last scored attempt for each task.
            for attempt in (candidate / split).glob("*/attempt-*"):
                cost["attempts"] += 1
                for status_name in ["feedback.json", "interrupted.json"]:
                    status_path = attempt / status_name
                    if status_path.exists():
                        cost["attempt_elapsed_sec"] += json.loads(
                            status_path.read_text()
                        ).get("elapsed_sec", 0)
                        break
                for calls in attempt.rglob("model_calls.jsonl"):
                    for line in calls.read_text().splitlines():
                        event = json.loads(line)
                        cost["model_requests"] += int(event.get("llm_request", False))
                        cost["model_elapsed_sec"] += (
                            event.get("duration_ms") or 0
                        ) / 1000
                        for key in ["input_tokens", "output_tokens"]:
                            cost[key] += (event.get("usage") or {}).get(key, 0)
                for calls in attempt.glob("nika/native/trials/*/tool_calls.jsonl"):
                    cost["executed_primitive_calls"] += sum(
                        not json.loads(line).get("denied")
                        for line in calls.read_text().splitlines()
                    )
            record["splits"][split]["measured_cost_including_retries"] = cost
    score = record["splits"].get("search", {}).get("mean_rca_f1")
    if score is not None:
        best = score if best is None else max(best, score)
    record["best_so_far_search_rca_f1"] = best
    record["hypothesis"] = next(
        (r.get("hypothesis") for r in rows if r["agent"] == name), "baseline"
    )
    records.append(record)
    curve.append(
        {
            "iteration": iteration,
            "candidate": name,
            "candidate_search_rca_f1": score,
            "best_so_far_search_rca_f1": best,
        }
    )
usage, requests, tools, durations = {}, 0, 0, 0.0
for path in root.rglob("model_calls.jsonl"):
    if "search_workspace" in path.parts:
        continue  # Those files are copies of the original evidence.
    for line in path.read_text().splitlines():
        event = json.loads(line)
        requests += int(event.get("llm_request", False))
        for key, value in (event.get("usage") or {}).items():
            if isinstance(value, (int, float)):
                usage[key] = usage.get(key, 0) + value
for path in root.glob(
    "candidates/*/*/*/attempt-*/nika/native/trials/*/tool_calls.jsonl"
):
    for line in path.read_text().splitlines():
        event = json.loads(line)
        tools += int(not event.get("denied"))
        durations += (event.get("duration_ms") or 0) / 1000
status = meta["status"]
if status != "complete":
    if (root / "frozen.json").exists():
        status = "selection_complete"
    elif (root / "search-complete.json").exists():
        status = "search_complete"
    elif rows:
        status = "searching"
report = {
    "experiment": str(root),
    "status": status,
    "upstream_commit": "8123ccabe2b19fa1123090b2e5f1bacc20963ce8",
    "nika_commit": meta["config"]["nika_commit"],
    "model": meta["config"]["model"],
    "curve": curve,
    "candidates": records,
    "model_requests_including_optimizer": requests,
    "usage": usage,
    "executed_primitive_calls": tools,
    "primitive_elapsed_sec": durations,
    "monetary_cost": None,
}
if (root / "frozen.json").exists():
    report["frozen"] = json.loads((root / "frozen.json").read_text())
a.output.mkdir(parents=True, exist_ok=True)
(a.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
lines = [
    "# Meta-Harness NIKA pilot results",
    "",
    f"Status: {status}. Model: `{meta['config']['model']}`. Backend: `{meta['config']['execution_mode']}`.",
    "",
    "| Iteration | Candidate | Search RCA F1 | Best search RCA F1 | Selection RCA F1 | Final RCA F1 |",
    "| ---: | --- | ---: | ---: | ---: | ---: |",
]


def fmt(value):
    return f"{value:.4f}" if value is not None else "Not evaluated"


for r in records:
    scores = [
        r["splits"].get(s, {}).get("mean_rca_f1")
        for s in ["search", "selection", "final"]
    ]
    lines.append(
        f"| {r['iteration']} | {r['candidate']} | {fmt(scores[0])} | {fmt(r['best_so_far_search_rca_f1'])} | {fmt(scores[1])} | {fmt(scores[2])} |"
    )
lines += [
    "",
    f"Measured model requests, including optimizer: {requests}. Executed primitive calls: {tools}. Primitive elapsed time: {durations:.1f} s. Token usage: `{json.dumps(usage, sort_keys=True)}`. Dollar cost is unknown.",
    "",
    "The best-so-far search score is nondecreasing by definition. Assess improvement using individual scores and the final held-out split.",
]
lines += [
    "",
    "Costs include interrupted and retried attempts. Model seconds measure proxy request duration; attempt seconds also include environment and sandbox preparation, execution and cleanup. They are not GPU compute seconds. Optimizer requests are included only in the total above.",
    "",
    "| Candidate | Split | Attempts | Model requests | Input tokens | Output tokens | Primitive calls | Model seconds | Attempt seconds |",
    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for r in records:
    for split, summary in r["splits"].items():
        c = summary["measured_cost_including_retries"]
        lines.append(
            f"| {r['candidate']} | {split} | {c['attempts']} | {c['model_requests']} | {c['input_tokens']} | {c['output_tokens']} | {c['executed_primitive_calls']} | {c['model_elapsed_sec']:.1f} | {c['attempt_elapsed_sec']:.1f} |"
        )
(a.output / "results.md").write_text("\n".join(lines) + "\n")
if curve:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r["iteration"] for r in curve]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        xs,
        [r["candidate_search_rca_f1"] for r in curve],
        marker="o",
        label="New candidate",
    )
    ax.step(
        xs,
        [r["best_so_far_search_rca_f1"] for r in curve],
        where="post",
        linestyle="--",
        label="Best so far",
    )
    ax.set(
        xlabel="Proposal iteration (0 = baseline)",
        ylabel="Mean native joint RCA F1",
        ylim=(-0.03, 1.03),
        xticks=xs,
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(a.output / "search-curve.svg")
    fig.savefig(a.output / "search-curve.png", dpi=160)
    plt.close(fig)
print(json.dumps({"status": status, "curve": curve, "model_requests": requests}))
