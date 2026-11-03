# Integrate a harness optimizer with NIKA

External optimizers own candidate generation, experiment history, search, and selection. NIKA owns the executable task, network lifecycle, sandbox, MCP access policy, model proxy, submission, and evaluation. The integration boundary is one complete task attempt.

## Execute a task attempt

The adapter submits a native `RolloutRequest` through `nika rollout run`. The request identifies the attempt, task, validated run configuration, and external agent project. NIKA registers that project in an output-local registry and runs its existing benchmark lifecycle. Each output directory is new.

`RolloutResult` separates the native outcome from a nullable terminal reward and references native artifacts. An agent failure receives zero; unresolved infrastructure errors receive null. The caller retains the task, candidate, attempt, session, configuration, model exchanges, primitive calls, submission, metrics, and cleanup records.

The learning adapter prioritizes native `infra_error` validity flags when scheduling retries and retains the original status as `native_outcome`. A [new validation run](../RUNBOOK.md#revalidate-a-completed-search) can reuse the entire searched cohort after a serving interruption. It retains candidate code and runtime identities, reevaluates selection, and keeps previous scores in the source experiment.

Native and Firecracker execution use the same request schema. NIKA prepares and restores snapshots, validates restored environments, and removes disposable VM copies. The learning controller does not create another emulator, scorer, or snapshot manager.

This whole-episode boundary follows the separation of agent and task execution in Terminal-Bench/Harbor and external predictions and evaluation in SWE-bench. NIKA continues to execute its own network tasks. See the native rollout reference for its Python and CLI contracts.

## Reuse the official search loop

The two vendored Meta-Harness files remain byte-identical to upstream commit `8123ccabe2b19fa1123090b2e5f1bacc20963ce8`. The adapter calls upstream `run_evolve`, frontier updates, iteration accounting, and evolution summaries. It replaces Harbor execution, validation, smoke evaluation, result parsing, and proposer transport with NIKA hooks.

The proposer uses bounded filesystem tools over a search-only workspace. It can page through prior candidate code and actual search interactions. Before writing a candidate or pending evaluation manifest, it must read historical code, search scores, and an execution trace. The controller records nonempty page reads per attempt. This is an evidence-access check, not a test of causal reasoning. It cannot execute shell commands or read selection/final data through these tools. Candidate code executes only inside the NIKA sandbox.

The proposer uses the configured Chat Completions service or the existing NIKA subscription Responses transport. Its context and accounting are separate from the evaluated agent. This is a network-domain adaptation of the upstream method. Different proposer tooling, continuous joint RCA F1, smaller task sets, and fixed serial execution distinguish it from the paper's numerical experiments.

## Evolve executable diagnosis code

Each candidate is a standalone module that returns a nonempty string. Static checks reject relative imports without executing candidate code on the controller. Runtime checks classify candidate import and report contract errors as agent failures while preserving native records. Explicit infrastructure flags take precedence.

Each candidate defines:

```python
class AgentHarness:
    async def run(self, task_description, ctx):
        message = await ctx.complete([
            {"role": "user", "content": task_description}
        ])
        return message.get("content") or "No diagnosis"
```

`ctx.tools` contains approved function schemas. `ctx.complete(messages, tools=None)` requests the fixed model. `ctx.call_tool(name, arguments)` executes an approved primitive and returns its observed output with an observation identity. `ctx.max_steps` and `ctx.model` expose fixed run settings. The maintained baseline demonstrates a full model/tool loop.

Candidates can implement wrappers, composites, result parsing, context selection, and model-based summaries. Each primitive still reaches the host MCP gateway, and each summary reaches the host model proxy. Host logs are authoritative. Candidate-produced summaries are not independent observations. The runtime uses a 600-second diagnosis timeout, at most 60 primitive calls, and the configured request cap. The final diagnosis request asks for a report using existing observations. Submission runs separately with a fixed worker.

The host enforces the selected access role and phase restrictions. The existing network shell capability can permit network mutations; diagnosis instructions alone are not a read-only security boundary. Keep the access policy fixed across candidates.

## Preserve only necessary identities

Source and contract digests prevent candidates, configuration, dependencies, or held-out partitions from changing during resume. Attempt manifests detect edits to archived interactions before cached feedback is reused. The native request digest identifies its serialized envelope. These checks protect comparison and recovery; the repository does not ship archive checksum bundles, historical patch bundles, or development snapshots.

The proposer workspace contains search records only. Reports and indices read original native records, excluding workspace copies from accounting. Tokens, model requests, primitive calls, and timing include failed and interrupted attempts. Unknown monetary prices remain null.

## Add training later

The existing interactions provide model inputs and outputs, tool messages, observations, phase identities, candidate source, outcomes, terminal reward, and task provenance for a future SFT converter. No converter or weight training is implemented.

RL integration still needs policy/checkpoint version management, a trainer sampling protocol, backpressure and concurrency control, behavior-policy metadata and log probabilities when required, a reward transport, and any reset/step or intermediate-state restoration contract. The current API provides whole episodes and terminal rewards.
