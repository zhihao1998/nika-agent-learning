# Validation evidence and limits

The automatic pipeline completed real network/model acceptance before source cleanup. The cleaned repository retains the executable path and durable regression tests. Raw archives remain on the experiment hosts, outside the source repository.

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

The cleaned controller currently passes 20 offline regression tests. The NIKA adaptation passes 110 focused tests. Ruff checks and formatting pass for maintained source files. These checks run on the cleanup revision, separately from the earlier live acceptance.

## Cleanup network integration check

A separate deterministic `mock` client executed one healthy `dc_clos` task through the new rollout CLI on xwangz-vm. Native outcome was success, reward 1.0, and all three primitive calls completed without errors. NIKA restored a real Firecracker network and completed submission and scoring. This client made no model requests and used the built-in test-agent path, so this is not new normal-sandbox/model acceptance or a performance measurement.

The executed native source was `61b3ede4`; later changes retain the full MCP protocol payload in host logs and correct documentation. Offline regressions cover those changes. The run is `/home/ubuntu/nika-harness-review-runs/attempt-01/`, with a local copy at `/workspace/nika-harness-review-archive/cleanup-native-episode/`. Its cleanup inspection found no running owned sessions or VM copies. The reusable prepared image remains cached.
