# NIKA Agent Learning

Run automatic harness engineering experiments on real NIKA network tasks with a fixed model. The built-in proposer writes diagnosis code from search feedback. NIKA executes each candidate in its normal sandbox and owns network environments, MCP permissions, model and tool budgets, submission, scoring, and cleanup.

The adapter runs the [official Meta-Harness](https://github.com/stanford-iris-lab/meta-harness) search loop. Candidate programs can change prompts, context management, diagnostic control flow, and tool wrappers. The operator configures experiments and reviews results; the proposer writes candidates.

## Run an experiment

You need Linux, Python 3.12 or later, [uv](https://docs.astral.sh/uv/), and a working [NIKA installation](https://github.com/sands-lab/nika) with the rollout API from [NIKA PR #307](https://github.com/sands-lab/nika/pull/307). Prepare the NIKA host and its sandbox backend before starting. Model serving remains a separate operator-managed service.

```sh
git clone git@github.com:zhihao1998/nika-agent-learning.git
cd nika-agent-learning
uv sync --frozen
uv run nika-learning --help
```

Follow the [runbook](RUNBOOK.md) to configure the native checkout and model, admit tasks, and execute:

```sh
uv run nika-learning init runs/study --config state/config.json \
  --tasks configs/tasks.json --admission state/admission
uv run nika-learning run runs/study --checkout /absolute/path/to/nika --until search
uv run nika-learning resume runs/study --checkout /absolute/path/to/nika --until final
uv run python scripts/report_meta_harness.py runs/study state/report
```

`state/config.json` must contain your NIKA commit, private state directory, model settings, and serving job identity. The runbook shows how to create it. The bundled tasks are public examples from earlier engineering runs; choose unseen tasks for a new independent evaluation.

## What stays fixed

The experiment records the task splits, NIKA commit, model settings, runtime, and candidate source. Only search traces reach the proposer. All candidates use the same execution and budget settings. Selection uses mean native joint RCA F1 and retains the earlier candidate on ties. Final evaluation compares baseline with the selected candidate after freezing its source.

Agent failures count as zero. Infrastructure failures have no reward and stop selection if bounded retries cannot resolve them. Completed trials remain cached within their own immutable attempt directories. Resume creates new attempts for interrupted work.

## Code and records

| Path | Purpose |
| --- | --- |
| `src/nika_learning/` | Official-loop adapter, bounded proposer, native execution, recovery |
| `meta_runtime/` | Fixed sandbox runtime and submission worker |
| `meta_candidates/baseline.py` | Initial executable harness |
| `configs/` | Service and subscription configuration templates, example task splits |
| `domain_spec.md` | Proposer instructions and candidate interface |
| `third_party/meta_harness/` | Pinned upstream source and license |
| `scripts/` | Task admission, Firecracker preparation, reporting, verification, resource inspection |
| `runs/` | Ignored experiment records, candidate code, actual model/tool exchanges, and scores |

Read [architecture](docs/architecture.md) for the NIKA boundary, upstream adaptation, data contracts, and training integration limits. Read [validation](docs/validation.md) for actual runtime evidence and its scope. Experiment archives and development history stay outside this source repository.

## Check changes

```sh
uv run pytest -q
uv run ruff check src meta_runtime meta_candidates scripts tests
uv run ruff format --check src meta_runtime meta_candidates scripts tests
```

The regression suite covers held-out isolation, provider protocols, budgets, candidate failure, freezing, and recovery. It runs offline. Real network and model execution require the host setup in the runbook.
