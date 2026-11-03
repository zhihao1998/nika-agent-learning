# Meta-Harness for NIKA

This experiment adapts the official Meta-Harness Terminal-Bench 2 search loop to network diagnosis. The base model stays fixed. The proposer evolves executable diagnosis programs from prior code, scores and full search traces. Before writing a candidate or proposal manifest, it must read nonempty pages of historical code, search scores, and execution traces. The controller archives those reads. This check establishes access to evidence, not whether the proposed change follows from it.

## Sources and adaptation boundary

The official repository is https://github.com/stanford-iris-lab/meta-harness, pinned at `8123ccabe2b19fa1123090b2e5f1bacc20963ce8`. The copied `meta_harness.py` and `claude_wrapper.py` remain byte-identical. The adapter executes the official `run_evolve` function, frontier updates, iteration accounting and evolution summary. NIKA hooks replace Harbor execution, agent validation, smoke evaluation, result parsing and the Claude proposer. This is a domain adaptation, not a reproduction of the paper's Terminal-Bench scores.

The native metric is continuous joint RCA F1. The adapter averages those scores instead of converting every positive reward into a pass. It uses one trial per task and one concurrent worker. The pilot uses a cheaper proposer model and fewer tasks and iterations than the paper. Each new candidate is evaluated independently; the frontier retains the best observed search score. A monotone frontier does not imply that successive candidates improve.

## Task and agent contract

A task identifies a real network topology, traffic pattern and optional injected fault. The agent observes the task description and approved MCP tools. It returns a diagnosis report. The fixed submission worker receives the report and NIKA's submission context, then invokes the normal submit tool. NIKA evaluates anomaly detection and exact resource/fault root causes. A healthy control requires an anomaly decision of false.

Each candidate is a standalone Python file in `agents/` defining `class AgentHarness` with `async run(task_description, ctx)`. Return a nonempty string. Relative imports are unsupported. `runtime-interface.py` documents the context; do not import it. The context provides:

- `ctx.tools`: approved OpenAI function schemas. A tool name is `schema["function"]["name"]`.
- `await ctx.complete(messages, tools=None)`: a typed assistant message from the fixed model.
- `await ctx.call_tool(name, arguments)`: a real primitive MCP observation with an observation identity.
- `ctx.max_steps` and `ctx.model`: fixed model request settings.

Candidates may change diagnostic control flow, prompts, context storage and retrieval, tool descriptions, schemas, aliases, parameter wrappers, composites, parsing and summaries. They may write executable Python using installed dependencies. Each candidate tests one mechanism supported by observed trajectories. They must preserve model tool-call/result pairing. They must not hardcode task names, topology/node names, fault locations or reference answers from history. Broad diagnostic principles apply across networks.

The host fixes the model, primitive permissions, phase boundaries, submission worker, tasks, scoring and budgets. Composite tools execute their actual primitives through MCP. The host gateway independently counts and records primitives across connections and server mounts. The model proxy counts all generation requests, including summaries, and rejects another model identity. Candidate code runs only in the normal sandbox. Syntax validation does not execute candidate code on the host. Host model and tool logs are authoritative; candidate messages are supplementary. Existing exec_shell permissions allow network writes, but candidates are instructed to diagnose without repair. No additional read-only enforcement is claimed.

## Experience and held-out data

The proposer receives a filesystem with all prior candidate code, search scores, native execution traces and reflections. Reads support arbitrary paging. Writes are limited to new candidate files and the pending evaluation manifest. Existing candidates and historical artifacts remain immutable. The proposer cannot access parent directories, selection or final data through its tools.

The frozen task manifest defines search, selection and final splits. Each split includes a healthy control. Only search feedback enters the proposer workspace. Selection and final tasks stay outside that workspace. Final rewards are viewed only after candidate freezing. Previously observed tasks provide engineering evidence, not new independent evidence.

All exact task definitions must pass NIKA's admission audit before initialization. Fault family and template annotations must not overlap across splits. Do not remove tasks because a harness cannot solve them. Agent failures count zero. Infrastructure failures have no reward, receive at most two attempts, and stop candidate selection if unresolved.

## Budget, selection and stopping

The frozen configuration defines proposal iterations, task counts and execution budgets. Evaluate baseline and candidates on the same search tasks. Model request and primitive limits apply across helper and summary calls. Valid zero scores are optimization feedback. Do not require a perfect baseline before proposing changes.

Selection evaluates all staged candidates on the fixed selection split using mean native joint RCA F1. Ties retain the earlier candidate, starting with baseline. The chosen source and experiment contract are frozen before final evaluation. Final evaluation compares baseline with the winner, executing an identical baseline/winner only once. No further generation follows selection. Search and held-out scores, requests, tokens, timings and failures are reported separately. Improvement is not an acceptance condition.

The automatic proposer owns candidate modifications. The operator configures and validates the pipeline. Do not manually edit staged candidates or replace their predictions.

## Execution backend and reusable interface

The NIKA `RolloutRequest` binds a task instance, agent settings, execution settings and request identity. `RolloutResult` separates execution outcome from a nullable reward and references native artifacts. Its API and `nika rollout run` CLI reuse the normal benchmark lifecycle. They contain no Meta-Harness path or candidate concept. Native and Firecracker execution share this contract. External search, sampling, SFT and RL frameworks can supply their own agent projects. The current interface executes whole episodes; it does not implement trainer transport or mid-trajectory policy updates.

This design follows Harbor's separation of task, agent, environment and verifier. It also follows SWE-bench's separation of a task identity, a submitted prediction and an independent evaluator. A fresh output directory prevents a new prediction from reusing a previous prediction's cached score. NIKA retains its own network task and submission semantics. This does not claim that NIKA directly executes SWE-bench or Terminal-Bench tasks.

Firecracker restores prepared lab and case images through NIKA's existing backend. The operator records image identity, restore time and owned VM cleanup. Baseline and candidates use the same backend and image settings. The runner must validate real restored tasks, not infer readiness from KVM availability. No weights are trained.

## Acceptance evidence

Save exact upstream and NIKA revisions, source hashes, configuration, candidate code, proposer interactions, task/attempt/session identities, actual model inputs and outputs, host tool calls and raw results, submissions, scores, failures and cleanup records. Run baseline and the configured proposal iterations on real tasks. Preserve interrupted attempts and verify recovery. Demonstrate a rejected over-budget primitive and a fixed-model request refusal. Plot new-candidate and best-so-far search scores separately. Freeze a winner before independent evaluation. Finish by checking only owned sessions, sandboxes and VM copies.
