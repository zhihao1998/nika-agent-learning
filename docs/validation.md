# Validation evidence and limits

The automatic pipeline completed real network/model acceptance before source cleanup. The cleaned repository retains the executable path and durable regression tests. Raw archives remain on the experiment hosts, outside the source repository.

## Qwen3-8B pilot and failure review

The November 3, 2026 study `meta-qwen3-8b-20261103` used a fixed Qwen3-8B checkpoint for both proposing and diagnosis. It ran the official loop through native NIKA `b75313bd`, controller `0228fb8`, normal agent sandboxes, and Firecracker on xwangz-vm. A separate three-request protocol check and a healthy sandbox episode completed before formal initialization. The sandbox episode submitted a valid but incorrect diagnosis, scored zero, and cleaned up.

| Stage | Observed result |
| --- | --- |
| Baseline search | Three tasks completed normally; joint RCA F1 was zero on all three |
| Proposal 1 | Evaluated on three tasks; returned a dictionary instead of a report string; all failed and scored zero |
| Proposal 2 | Rejected for an unterminated string literal; search continued |
| Proposal 3 | Relative imports failed before diagnosis; native classified the failure as infrastructure; two attempts exhausted and search stopped |
| Proposal 4, selection, final | Not executed |

The proposer made three proposals but read only the domain specification and runtime reference. It did not read historical candidate code, scores, or interactions. The interface made those records available, but the run did not demonstrate feedback-driven improvement. Both fault-task baseline diagnoses stopped after one pingmesh call and guessed routing or BGP causes, missing a down link and a missing host address. The healthy baseline falsely reported an ACL fault. Candidate implementation failures prevented a valid improvement comparison. These observations do not establish that small-model harness optimization is ineffective.

The formal partial run recorded 47 model requests (20 agent and 27 optimizer), 229,832 input tokens, 31,378 output tokens, and 14 primitive calls. This excludes the separate preflight and bootstrap checks. The main serving allocation lasted 2,659 seconds on two V100 GPUs, approximately 1.48 allocated GPU-hours. The controller released its job and the resource audit found no owned sessions, sandboxes, workspaces, or VM copies. Artifact verification reported no integrity errors. These checks establish provenance and cleanup, not efficacy.

Raw source records remain outside Git: xwangz-vm `/home/ubuntu/nika-agent-learning-validation/runs/meta-qwen3-8b-20261103/`, and nika01 `state/qwen3-8b-20261103/` in the validation worktree. Original attempts and scores remain unchanged.

## Candidate and feedback regression checks

The failure review added an evidence-read gate before candidate and proposal writes, shared syntax/interface checks, rejection of relative imports, and explicit runtime import/report contract errors. The runtime reference now exposes the context class without copying internal package imports. Explicit native infrastructure flags still produce a null reward.

All 37 offline tests passed locally after these fixes. They cover empty-page rejection, required historical code/score/trace reads, candidate return types, candidate import errors, infrastructure precedence, and continuation after a rejected relative import through the official two-round loop. Ruff checks and formatting passed. Candidate code executes only inside the normal sandbox during experiments; controller validation uses Python syntax trees. These are regression checks, not a new real-model search or held-out evaluation. A fresh experiment is required to test effectiveness with the changed runtime.

## Shared serving recovery checks

On November 3, 2026, all 30 offline tests passed after adding frozen-cohort validation, serving-load checks, and infrastructure-flag classification. Tests cover refusing busy serving before creating an attempt, cleaning up when contention appears during execution, and preserving candidate code and source records while reevaluating an entire cohort. The frozen-cohort path does not call the optimizer or import earlier scores.

A regression check also read the archived Qwen3.8 pilot's real 503 attempt. The adapter classified it as infrastructure failure with no reward, preserved native `agent_failed`, and left the old archive unchanged. All five candidate projects from that pilot copied byte-for-byte into a separate initialization check. These checks made no model requests or network rollouts; they do not establish a completed independent evaluation or performance improvement.

## Real automatic search acceptance

The Qwen3.6-35B-A3B-FP8 run `meta-qwen36-pipeline-smoke-20261102` completed baseline and two automatically generated candidates on two search tasks, totaling six native scored Firecracker/sandbox trials. The second proposer read the first candidate's source, scores, and actual interactions. Both candidate files matched the model's write operations. Completed resume created no new attempts or model requests.

Search means were 1.0, 0.83335, and 1.0. This validates the loop, including worse and tied candidates; it does not demonstrate improvement. Selection and final evaluation did not run in this Qwen acceptance. Its learning source was `8a181f2`, native NIKA was `96a3e6bd`, and the official loop was `8123ccab`.

The run recorded 63 agent requests and 27 proposer requests, 1,444,092 input tokens and 57,888 output tokens. The separately counted API checks used three requests. The two-GPU serving allocation lasted 1,763 seconds, approximately 0.9794 allocated GPU-hours. The owned serving job was canceled and resource inspection found no remaining owned sessions, sandboxes, workspaces, or VM copies.

Source records:

- xwangz-vm: `/home/ubuntu/nika-agent-learning-staged/runs/meta-qwen36-pipeline-smoke-20261102/`.
- nika01: `/workspace/nika-harness-review-archive/PIPELINE-VALIDATION.md` and `meta-qwen36-pipeline-smoke-20261102.tar.gz` in that archive directory.
- The earlier Luna pilot exercised selection, freezing, independent evaluation, candidate failure, and interrupted recovery. Its original report and raw run archive are retained in the same archive directory.

Historical task observations are engineering evidence. The bundled example tasks are not a fresh independent test set. The stopped 0.2.0 expanded experiment did not establish a quality improvement.

## Cleanup review

The source release removes the earlier configuration-only optimizer/runtime, duplicated CLI, staged and release-study controllers, machine-specific configuration variants, historical integration bundles, raw results, and one-use diagnostic scripts. Reporting, task admission, snapshot preparation, recovery, source integrity, and native trace verification remain.

The review corrected backend readiness checks, native-mode verification, report status, stale task admissions, and proposal-failure handling. Search history now excludes private execution requests and task injection configuration. It replaced policy-wrapper scaffolding in the fixed runtime with direct approved primitive execution. Candidate programs own tool wrappers and orchestration. Source/contract and sealed-attempt checks remain because they protect resume and score attribution.

The retained offline tests exercise the official loop for two rounds, invalid-candidate continuation, selection ties, freezing, final evaluation, idempotent resume, interrupted attempt preservation, provider protocol pairing, primitive metering, and held-out/path isolation. Native focused tests cover rollout outcomes and output reuse, remote access policies, model/primitive budgets, external manifests, Firecracker preparation, and container lookup.

The source-cleanup revision passed 20 offline regression tests. The NIKA adaptation passes 110 focused tests. Ruff checks and formatting pass for maintained source files. These checks run on the cleanup revision, separately from the earlier live acceptance.

## Cleanup network integration check

A separate deterministic `mock` client executed one healthy `dc_clos` task through the new rollout CLI on xwangz-vm. Native outcome was success, reward 1.0, and all three primitive calls completed without errors. NIKA restored a real Firecracker network and completed submission and scoring. This client made no model requests and used the built-in test-agent path, so this is not new normal-sandbox/model acceptance or a performance measurement.

The executed native source was `61b3ede4`; later changes retain the full MCP protocol payload in host logs and correct documentation. Offline regressions cover those changes. The run is `/home/ubuntu/nika-harness-review-runs/attempt-01/`, with a local copy at `/workspace/nika-harness-review-archive/cleanup-native-episode/`. Its cleanup inspection found no running owned sessions or VM copies. The reusable prepared image remains cached.
