# Run and resume a harness experiment

Use this runbook on a prepared NIKA Linux host. Run commands from this source checkout. `uv sync --frozen` installs the controller; it does not provision Docker, sandbox execution, Firecracker, or a model server.

## Prepare NIKA and the model

Use a separate NIKA checkout with `nika rollout run` and a working private `.venv`. Install NIKA with its normal installer and verify one ordinary task first. Keep that checkout clean throughout the experiment.

```sh
export NIKA_CHECKOUT=/absolute/path/to/nika
uv sync --frozen
mkdir -p state
cp configs/service.json state/config.json
```

For an existing NIKA-supported subscription login, copy `configs/subscription.json` instead. Credentials stay in the normal NIKA login store. Never put API keys in experiment JSON or candidate source. Subscription Responses requests use the fixed low reasoning effort; this transport does not enforce the configured output-token cap.

The service template uses the existing mcmgt01 load balancer. Start a model through its documented Slurm serving workflow. Inspect `/v1/models` and `/status`; wait until the selected model is ready. Set the exact serving job IDs before initialization. The controller never starts, replaces, or cancels a model job.

Fill in the configuration, using your actual job ID for a service-backed run:

```sh
uv run python - <<'PY'
import json, os, subprocess
from pathlib import Path
p = Path('state/config.json')
cfg = json.loads(p.read_text())
cfg['nika_commit'] = subprocess.check_output(
    ['git', '-C', os.environ['NIKA_CHECKOUT'], 'rev-parse', 'HEAD'], text=True
).strip()
cfg['state_dir'] = str(Path('state/nika').resolve())
# Service-backed runs: replace this empty list with actual ready job IDs.
# cfg['serving_job_ids'] = ['YOUR_JOB_ID']
p.write_text(json.dumps(cfg, indent=2) + '\n')
PY
```

| Setting | Meaning |
| --- | --- |
| `nika_commit` | Exact clean native checkout to execute |
| `model`, `model_api`, `provider`, `base_url` | Fixed serving model and transport |
| `serving_job_ids` | Required allowlist of ready backends for the mcmgt01 service |
| `state_dir` | Absolute experiment-owned native state directory |
| `execution_mode`, `execution_options` | Standard NIKA native or Firecracker settings |
| `rounds` | Positive bounded proposal count |
| `max_steps`, `optimizer_requests` | Model request limits per agent phase and proposal attempt |
| `underlying_tool_calls` | Host diagnosis primitive cap, at most the runtime's 60 |
| `agent_timeout_sec`, `case_timeout_sec` | Agent and complete native task deadlines |
| `total_timeout_sec` | Experiment wall budget, including pauses and preparation after init |
| `max_attempts` | Maximum attempts for a task or proposal round |

The fixed diagnosis runtime has a 600-second timeout. Submission has the same request cap as diagnosis and a separate 20-primitive cap. Set the serving lease and total budget to cover the entire planned run. Do not edit frozen settings after initialization.

## Admit your task splits

`configs/tasks.json` is an eight-task engineering example: two search tasks, three selection tasks, and three final tasks. Its tasks have already been observed in development. For new research, provide your own manifest and admit it on the exact NIKA commit you will execute.

Each entry contains a unique path-safe `id`, a `family`, a `template`, and a native task dictionary under `case`. Each split needs a healthy control. Fault families, templates, and exact task instances must not cross splits. Task annotations remain the operator's responsibility. The proposer receives search feedback only. Keep reference answers outside the sandbox project.

Run the existing native audit with private state:

```sh
export NIKA_STATE_DIR="$(uv run python -c 'import json; print(json.load(open("state/config.json"))["state_dir"])')"
PYTHONPATH="$PWD/src:$NIKA_CHECKOUT/src" "$NIKA_CHECKOUT/.venv/bin/python" \
  scripts/audit_tasks.py configs/tasks.json state/admission
```

Admission creates real network resources through NIKA and can take time. Inspect `state/admission/report.json`. Every task must pass. Records bind the native commit and task definition; failed or stale admission blocks initialization. Preserve failures instead of removing agent-unsolved tasks.

## Use Firecracker snapshots

Complete NIKA's Firecracker setup first. Configure a private image root before preparing tasks. For example, modify `state/config.json` before init:

```json
{
  "execution_mode": "firecracker",
  "execution_options": {
    "firecracker": {
      "root": "/absolute/experiment/firecracker",
      "case_images": true
    }
  }
}
```

These fields are an excerpt to merge into the full config. Configure native CPU, memory, huge pages, and image preparation through NIKA. For an existing approved image root, inspect its current images before reuse; do not remove shared caches.

Prepare native settings from the experiment template:

```sh
uv run python - <<'PY'
import json, yaml
from pathlib import Path
cfg = json.loads(Path('state/config.json').read_text())
Path('state/native.yaml').write_text(yaml.safe_dump({
    'nika': {'execution': {'mode': cfg['execution_mode'], **cfg['execution_options']}}
}))
PY
export NIKA_RUN_CONFIG="$PWD/state/native.yaml"
PYTHONPATH="$PWD/src:$NIKA_CHECKOUT/src" "$NIKA_CHECKOUT/.venv/bin/python" \
  scripts/prepare_rollouts.py configs/tasks.json state/preparation.json
```

Preparation uses NIKA's own image manager and verifies requested images. Use the same backend and image settings for baseline and candidates.

## Search, select, and evaluate

```sh
uv run nika-learning init runs/study --config state/config.json \
  --tasks configs/tasks.json --admission state/admission
uv run nika-learning run runs/study --checkout "$NIKA_CHECKOUT" --until search
uv run nika-learning resume runs/study --checkout "$NIKA_CHECKOUT" --until selection
uv run nika-learning resume runs/study --checkout "$NIKA_CHECKOUT" --until final
```

Search runs baseline and the configured proposal rounds. Each proposal reads previous search evidence and writes executable candidate code. The first search task also acts as its sandbox check; the full batch reuses that task's completed result.

Selection evaluates staged candidates on selection tasks. Mean native joint RCA F1 determines the winner; ties retain the earlier candidate, beginning with baseline. `frozen.json` records selection before final evaluation. Final runs baseline and winner on the final split; if baseline wins, it executes once. Stop after search if you only need to verify the automatic loop.

## Interrupt and resume

Send SIGINT or SIGTERM to the foreground controller. Wait for native cleanup to finish. Then resume the same experiment with the same source, native checkout, and configuration:

```sh
uv run nika-learning resume runs/study --checkout "$NIKA_CHECKOUT" --until final
```

The controller lock prevents simultaneous controllers. Completed attempts stay cached. Interrupted attempts retain their records, and recovery starts a new attempt. Invalid candidate code counts as a failed candidate and permits the next round. A failed proposer stops explicitly; resume retries that round within `max_attempts`. Unresolved infrastructure failures prevent selection.

Paused time consumes the experiment's wall budget. Resume does not extend an expired deadline or replace frozen serving jobs. A killed host, surviving orphan workers, mid-trajectory continuation, and automatic serving lease renewal require additional recovery work. Inspect and close recorded owned sessions before restarting after an ungraceful crash.

## Inspect records and cleanup

After the controller exits:

```sh
uv run python scripts/report_meta_harness.py runs/study state/report
uv run python scripts/index_meta_runs.py runs/study state/report/attempts.csv
uv run python scripts/verify_meta_harness.py runs/study state/verification.json
uv run python scripts/audit_meta_resources.py state/resources.json \
  --run runs/study --profile "$NIKA_CHECKOUT" state/config.json
```

`results.json` and `results.md` separate search, selection, and final scores and report cost including retries. The plot separates candidate scores from best-so-far search scores. `attempts.csv` links each score to its native session, model/tool calls, submission, metrics, and run identity. Verification checks source integrity, score attribution, budgets, and task/configuration pairing.

The resource inspector does not delete resources. Close recorded experiment-owned sessions with NIKA lifecycle commands. Cancel only a Slurm model job you created after all runs are finished. Keep unrelated sandboxes, sessions, model jobs, and reusable caches.
