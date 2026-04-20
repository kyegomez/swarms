# Swarms Multi-Agent Architecture: Performance Improvement Ideas

A comprehensive catalog of performance, reliability, and feature improvement ideas for the core multi-agent structures in the Swarms framework.

---

## Table of Contents

- [SequentialWorkflow](#sequentialworkflow)
- [ConcurrentWorkflow](#concurrentworkflow)
- [HierarchicalSwarm](#hierarchicalswarm)
- [AgentRearrange](#agentrearrange)
- [GraphWorkflow](#graphworkflow)
- [Cross-Cutting Improvements](#cross-cutting-improvements)
- [GitHub Issues Tracking](#github-issues-tracking)

---

## SequentialWorkflow

**File:** `swarms/structs/sequential_workflow.py`
**Engine:** `swarms/structs/agent_rearrange.py`

### Context & Memory

| # | Improvement | Description |
|---|-------------|-------------|
| 1 | **Sliding window context** | Instead of passing the full conversation history to every agent via `get_str()`, pass only the last N messages or a summary. Token usage currently grows quadratically — agent 5 sees all prior outputs. |
| 2 | **Context summarization between steps** | Insert an automatic summarizer agent that compresses prior outputs before handing off, reducing token bloat in long chains. |
| 3 | **Cache `get_str()` output** | The conversation is serialized to string on every agent call. Cache the result and only rebuild the delta. *(GitHub Issue: [#1460](https://github.com/kyegomez/swarms/issues/1460))* |

### Execution

| # | Improvement | Description |
|---|-------------|-------------|
| 4 | **Pipeline parallelism** | When running batched tasks, agent 2 could start on task 1's output while agent 1 processes task 2. Currently tasks are fully serialized. |
| 5 | **Async-native execution** | `run_async` currently wraps synchronous `run()` in `asyncio.to_thread`. A true async path with `await agent.arun()` would reduce thread overhead. |
| 6 | **Early termination** | Add a confidence/quality check between steps so the chain can stop early if output is already sufficient (e.g., skip the reviewer if the writer's output scores high). |

### Overhead Reduction

| # | Improvement | Description |
|---|-------------|-------------|
| 7 | **Cache flow validation** | `validate_flow()` runs on every call with O(n) agent name checks. Parse and validate once at init, invalidate only on flow change. *(GitHub Issue: [#1461](https://github.com/kyegomez/swarms/issues/1461))* |
| 8 | **Lazy output formatting** | `history_output_formatter()` processes the entire history at the end. For `"final"` output type, skip serializing everything and just return the last message. |
| 9 | **Eliminate redundant `any_to_str()` calls** | Every agent response goes through `any_to_str()` conversion even when it's already a string. *(GitHub Issue: [#1462](https://github.com/kyegomez/swarms/issues/1462))* |

### Reliability

| # | Improvement | Description |
|---|-------------|-------------|
| 10 | **Checkpoint and resume** | Save conversation state after each agent step so a failed workflow can resume from the last successful agent instead of restarting from scratch. |

---

## ConcurrentWorkflow

**File:** `swarms/structs/concurrent_workflow.py`

### Thread & Execution Model

| # | Improvement | Description |
|---|-------------|-------------|
| 1 | **Async with `asyncio`** | Replace `ThreadPoolExecutor` with native async execution. LLM API calls are I/O-bound, making async ideal. |
| 2 | **Adaptive worker pool sizing** | Currently hardcoded at 95% of CPU cores. For 3 agents that's wasteful; for 200 agents it's a bottleneck. Scale workers to `min(agent_count, cpu_cores)`. |
| 3 | **Connection pooling per LLM provider** | Multiple agents hitting the same API endpoint should share HTTP connections to reduce TLS handshake overhead. |

### Result Handling

| # | Improvement | Description |
|---|-------------|-------------|
| 4 | **Stream-first results** | Use `as_completed()` in dashboard mode too (currently uses `wait()` which blocks until the slowest agent finishes). Return partial results as each agent completes. |
| 5 | **Result streaming to disk** | For large outputs from many agents, stream results to disk instead of holding everything in memory. Current approach holds all outputs in RAM. |
| 6 | **Parallel result aggregation** | Adding results to the conversation happens sequentially after all agents finish. Add them as each future completes. |

### Dashboard & Monitoring

| # | Improvement | Description |
|---|-------------|-------------|
| 7 | **Reduce dashboard refresh overhead** | The 0.1s throttle on dashboard updates causes ~100+ dict mutations/sec on large swarms. Batch status updates and refresh at 1-2 FPS instead. |
| 8 | **Decouple dashboard from execution thread** | Run dashboard rendering in a separate thread so it never blocks agent execution. |

### Batching

| # | Improvement | Description |
|---|-------------|-------------|
| 9 | **Parallel batch execution** | `batch_run()` processes tasks sequentially, running agents concurrently only within each task. Process multiple tasks concurrently too. |
| 10 | **Rate-limit-aware scheduling** | Add API rate limit awareness so the executor can throttle submissions per provider instead of hitting 429s and retrying. |

### Error Handling

| # | Improvement | Description |
|---|-------------|-------------|
| 11 | **Timeout per agent** | No per-agent timeout exists. A single hanging agent blocks the entire workflow forever. Add configurable per-agent timeouts with `future.result(timeout=X)`. |
| 12 | **Retry with backoff** | Failed agents (especially on transient API errors) should retry with exponential backoff before reporting failure. |

---

## HierarchicalSwarm

**File:** `swarms/structs/hierarchical_swarm.py`

### Director Optimization

| # | Improvement | Description |
|---|-------------|-------------|
| 1 | **Lightweight director model** | The director just assigns tasks; it doesn't need the same heavy model as workers. Use a faster/cheaper model (e.g., Haiku) for task distribution. |
| 2 | **Skip planning phase when unnecessary** | `planning_enabled=True` adds a full extra LLM call. For simple tasks or single-loop runs, skip it automatically. |
| 3 | **Cache director decisions** | For repeated similar tasks, cache the director's task distribution plan and reuse it. |
| 4 | **Batch director calls across loops** | In multi-loop mode, the director could plan multiple iterations ahead instead of one loop at a time. |

### Agent Execution

| # | Improvement | Description |
|---|-------------|-------------|
| 5 | **O(1) agent lookup** | `call_single_agent` does O(n) linear search for agents by name. Use a dict keyed by `agent_name` — the sequential/concurrent workflows already do this. *(GitHub Issue: [#1458](https://github.com/kyegomez/swarms/issues/1458))* |
| 6 | **Parallel execution by default** | Sequential is the default but most hierarchical tasks are independent. Make parallel the default and fall back to sequential only when dependencies exist. *(GitHub Issue: [#1458](https://github.com/kyegomez/swarms/issues/1458))* |
| 7 | **Dependency-aware scheduling** | Add a DAG-based scheduler so the director can specify dependencies between orders. Independent tasks run in parallel; dependent ones wait. |
| 8 | **Per-agent timeout and cancellation** | In parallel mode, `as_completed()` waits for all futures. Add timeouts so a slow agent doesn't block the whole swarm. |

### Feedback Loop

| # | Improvement | Description |
|---|-------------|-------------|
| 9 | **Conditional feedback** | Run the judge/feedback agent only when output quality is uncertain. If all agent outputs exceed a quality threshold, skip the feedback loop and save an LLM call. |
| 10 | **Incremental feedback** | Instead of reviewing all outputs at once, provide feedback as each agent completes. This lets later agents benefit from earlier feedback within the same loop. |
| 11 | **Reduce feedback agent context** | The feedback director receives the full conversation. Pass only the current loop's outputs + the original task. |

### Context Management

| # | Improvement | Description |
|---|-------------|-------------|
| 12 | **Conversation pruning between loops** | History grows with every loop. Summarize or prune earlier loops' outputs to stay within token limits. Currently risks exceeding context windows on multi-loop runs. |
| 13 | **Selective context per agent** | Not every worker needs the full conversation. Pass only the director's order + relevant prior outputs instead of `conversation.get_str()`. |

### Structural

| # | Improvement | Description |
|---|-------------|-------------|
| 14 | **Sub-hierarchies** | Allow a worker agent to itself be a HierarchicalSwarm, enabling tree-structured task decomposition for complex problems. |
| 15 | **Agent pool with auto-selection** | Instead of the director manually naming agents, let it describe the capability needed and auto-match to the best available agent. See [implementation options below](#agent-pool-auto-selection-options). |

#### Agent Pool Auto-Selection Options

Five approaches for matching capability descriptions to agents, in order of increasing complexity:

**Option A — Tag-Based Matching** *(recommended starting point)*
Each agent declares skill tags at init. The director outputs required tags instead of names. Matching is pure set intersection — no ML needed, fully deterministic.
```
agent = Agent(tags=["finance", "risk-assessment", "quantitative"], ...)
order.required_tags = ["finance", "risk-assessment"]
# score = len(required_tags & agent.tags)
```
- Pros: Fast, debuggable, zero dependencies.
- Cons: Rigid vocabulary — director and agents must share the same tag set.

**Option B — LLM-as-Router** *(most backward compatible)*
Keep the existing `agent_name` field in `HierarchicalOrder`. If the name doesn't match any registered agent, fire a lightweight LLM call to resolve the capability description to the best matching agent name.
```python
def resolve_agent(self, order):
    if order.agent_name in self.agent_map:
        return order.agent_name  # fast path — exact match
    # slow path — LLM picks best agent from pool
    return self.director.run(f"Which agent best handles: {order.agent_name}?")
```
- Pros: Zero schema changes, backward compatible.
- Cons: Extra LLM call on every mismatch; potential hallucinations.

**Option C — Embedding-Based Semantic Matching**
Compare a required-capability string against each agent's `agent_description` / `system_prompt` using cosine similarity of embeddings. Most flexible — handles fuzzy or novel requests that don't fit predefined tags.
```python
query_vec = embed(order.required_capability)
best = max(self.agents, key=lambda a: cosine_sim(query_vec, embed(a.agent_description)))
```
- Pros: Handles natural language capability descriptions accurately.
- Cons: Requires an embedding model; adds latency per routing decision.

**Option D — Capability Registry with Proficiency Scores**
Agents declare structured capabilities with 0–1 proficiency scores. The director requests a skill + minimum proficiency. Matching is a ranked lookup returning the most proficient available agent.
```python
class AgentCapability(BaseModel):
    skill: str
    proficiency: float  # 0.0 - 1.0

# Routing: filter by skill + min_proficiency, then rank by score
```
- Pros: Fine-grained control, supports load balancing (use second-best if best is busy).
- Cons: Agents must self-declare proficiency, which can be inaccurate or stale.

**Option E — Hybrid Tags + Embedding Fallback** *(best accuracy overall)*
Fast tag intersection for the common case; semantic embedding fallback when tag confidence is low.
```python
if max_tag_score >= 2:
    return tag_match   # confident
return embedding_match(order.capability_description)  # fallback
```
- Pros: Fast in the common case, graceful on edge cases.
- Cons: Two systems to maintain; requires embedding dependency.

| Option | Latency | Accuracy | External Deps | Backward Compatible | Complexity |
|--------|---------|----------|---------------|---------------------|------------|
| A. Tag matching | Very low | Medium | None | No | Low |
| B. LLM-as-router | Low (hit) / High (miss) | High | None | Yes | Low |
| C. Embedding match | Medium | High | Embedding model | No | Medium |
| D. Capability registry | Very low | High | None | No | Medium |
| E. Hybrid tags + embed | Low / Medium | Highest | Embedding model | No | High |

---

## AgentRearrange

**File:** `swarms/structs/agent_rearrange.py`

### Context & Token Efficiency

| # | Improvement | Description |
|---|-------------|-------------|
| 1 | **Quadratic token growth** | Every agent receives the full conversation via `get_str()` (lines 509, 572). With 5 agents each producing 1K tokens, agent 5 processes ~5K input tokens. Add a sliding window or summary mode. |
| 2 | **Cache `get_str()` result** | `conversation.get_str()` rebuilds the full string on every call. Cache it and invalidate only when new messages are added. *(GitHub Issue: [#1460](https://github.com/kyegomez/swarms/issues/1460))* |
| 3 | **Selective context per agent** | Not every agent needs all prior outputs. Allow the flow syntax to specify which upstream outputs an agent should see (e.g., `agent3[agent1]` = only receive agent1's output). |

### Validation & Parsing Overhead

| # | Improvement | Description |
|---|-------------|-------------|
| 4 | **Parse flow once, not per-run** | `validate_flow()` and `self.flow.split("->")` re-parse the flow string on every `_run()` call. Parse into a structured execution plan once at init. *(GitHub Issue: [#1461](https://github.com/kyegomez/swarms/issues/1461))* |
| 5 | **Drop redundant `any_to_str()` calls** | Line 577 runs `any_to_str(current_task)` on every agent response. If `agent.run()` already returns a string, this is wasted work. Guard with `isinstance` check first. *(GitHub Issue: [#1462](https://github.com/kyegomez/swarms/issues/1462))* |

### Execution Model

| # | Improvement | Description |
|---|-------------|-------------|
| 6 | **True async execution** | `run_async` wraps synchronous `run()` in `asyncio.to_thread`. A native async path using `await agent.arun()` would avoid thread pool overhead. |
| 7 | **Pipeline parallelism for `batch_run`** | `batch_run` processes tasks sequentially. Agent 2 could start on task 1 while agent 1 works on task 2 — free latency savings. *(GitHub Issue: [#1463](https://github.com/kyegomez/swarms/issues/1463))* |
| 8 | **Fix `batch_run` to use `ThreadPoolExecutor`** | The comment says "process batch using concurrent execution" but it's a sequential list comprehension. *(GitHub Issue: [#1463](https://github.com/kyegomez/swarms/issues/1463))* |

### Error Handling & Resilience

| # | Improvement | Description |
|---|-------------|-------------|
| 9 | **`_run()` swallows exceptions** | `_catch_error` logs and returns the exception but `_run` discards the return value, silently returning `None` on failure. *(GitHub Issue: [#1464](https://github.com/kyegomez/swarms/issues/1464))* |
| 10 | **`run()` double-logs on failure** | `run()` → `_run()` → `_catch_error()`, then `run()` calls `_catch_error()` again. Two telemetry writes and two log entries for the same error. *(GitHub Issue: [#1465](https://github.com/kyegomez/swarms/issues/1465))* |
| 11 | **No per-agent timeout** | A single hanging agent blocks the entire workflow forever. Add configurable per-step timeouts. |
| 12 | **Checkpoint & resume** | If agent 4 of 5 fails, the entire workflow restarts. Save conversation state after each step and resume from the last successful agent. |

### Telemetry Overhead

| # | Improvement | Description |
|---|-------------|-------------|
| 13 | **`to_dict()` called twice per run** | `run()` calls `log_agent_data(self.to_dict())` both before and after execution. Move telemetry to a background thread or make it opt-in. |
| 14 | **`to_dict()` serialization cost** | Iterates all `__dict__` items and attempts `json.dumps()` on each to test serializability — O(n) per attribute, twice per run. Cache or lazily serialize. |

### Concurrency Safety

| # | Improvement | Description |
|---|-------------|-------------|
| 15 | **Race condition in concurrent workflow** | `_run_concurrent_workflow` has agents read `self.conversation.get_str()` and write back to `self.conversation` without locking. Snapshot the conversation before concurrent execution. |
| 16 | **Shared conversation across `concurrent_run`** | `concurrent_run()` runs multiple tasks on the same instance sharing `self.conversation`. Parallel tasks corrupt each other's history. Each task needs its own conversation copy. |

### Flow Syntax & Flexibility

| # | Improvement | Description |
|---|-------------|-------------|
| 17 | **Conditional branching** | Add support for conditional routing where the next agent depends on the previous output. See [implementation options below](#conditional-branching-options). |
| 18 | **Agent repetition in flow** | Formally support loops like `agent1 -> agent2 -> agent1` for iterative refinement. *(GitHub Issue: [#1466](https://github.com/kyegomez/swarms/issues/1466))* See [implementation options below](#agent-repetition-options). |
| 19 | **Weighted concurrent execution** | When agents run concurrently there's no priority. Allow weighting so critical agents get thread priority or earlier result collection. See [implementation options below](#weighted-concurrent-execution-options). |

---

#### Conditional Branching Options

**Option A — Ternary Syntax in Flow String**
Extend the flow DSL with `?` (branch) and `:` (else):
```
"analyst -> reviewer ? writer : editor -> publisher"
# After analyst, if reviewer approves → writer, else → editor. Both converge at publisher.
```
The executor replaces the flat list iteration with a graph walker. Conditions are evaluated via one of three strategies:

| Strategy | Syntax example | How it works |
|----------|---------------|--------------|
| Keyword match | `reviewer?contains(approved)` | `"approved" in result.lower()` |
| LLM-as-judge | `reviewer?` | Small LLM call returning yes/no |
| User callable | `condition_fns={"check": lambda r: ...}` | Referenced by name in flow string |

- Pros: Declarative, visible in the flow string, composable with existing `->` and `,` syntax.
- Cons: Parser complexity increases; requires moving from a flat list to a tree/graph structure.

**Option B — RouterAgent** *(recommended starting point)*
No new syntax. A built-in `RouterAgent` reads the prior output and returns the name of the next agent to invoke. The executor detects router outputs and dynamically dispatches:
```python
flow = "analyst -> router -> publisher"
router = RouterAgent(
    candidates={"writer": "creative tasks", "editor": "fix/polish tasks"},
    routing_strategy="llm"  # or "keyword" or "callable"
)
```
- Pros: Zero flow syntax changes, fully backward compatible, ships immediately.
- Cons: Branching logic is invisible in the flow string — not declarative.

**Option C — DAG-Based Flow** *(most powerful)*
Replace the string flow entirely with a graph object:
```python
graph = FlowGraph()
graph.add_edge("analyst", "reviewer")
graph.add_edge("reviewer", "writer",  condition=lambda r: "approved" in r)
graph.add_edge("reviewer", "editor",  condition=lambda r: "rejected" in r)
graph.add_edge("writer",   "publisher")
graph.add_edge("editor",   "publisher")

rearrange = AgentRearrange(agents=[...], flow=graph)
```
Supports any topology: branching, merging, cycles, parallel branches. String syntax becomes sugar that compiles to the graph.

- Pros: Maximum expressiveness; handles any workflow topology.
- Cons: Largest breaking change; loses simplicity of the string DSL.

**Recommendation:** Ship Option B (RouterAgent) first for zero-risk delivery. Design Option C (DAG) as the long-term architecture, with Option A string syntax as a layer on top.

---

#### Agent Repetition Options

Currently `response_dict[agent_name] = result` overwrites prior results from the same agent and `_get_sequential_awareness` finds only the first occurrence of a name.

**Option A — Position-Indexed Execution** *(recommended)*
Track position in the flow, not just agent name. Three focused fixes:
```python
# 1. Use index in result storage (no overwrite)
response_dict[f"{agent_name}_step_{task_idx}"] = result

# 2. Pass position to awareness lookup
def _get_sequential_awareness(self, agent_name, tasks, current_position):
    # Use current_position instead of searching for first occurrence

# 3. Remove the commented-out duplicate agent check (lines 330-333)
```
- Pros: Minimal code change — fixes three specific bugs without redesigning anything.
- Cons: Result dict keys are less ergonomic (`writer_step_0`, `writer_step_2`).

**Option B — Repeat Count Syntax**
Add `*N` and `[block]*N` syntax that expands at parse time:
```
"writer -> reviewer -> writer*3"              # writer runs 3 times
"[writer -> reviewer]*2 -> publisher"         # block repeats twice
```
Parser expands before execution — the runtime never sees repetition, just a longer flat plan.
```python
# "writer*3"         → "writer -> writer -> writer"
# "[A -> B]*2"       → "A -> B -> A -> B"
```
- Pros: Compact and readable in the flow string; runtime unchanged.
- Cons: Expansion happens before context-awareness, so each repeated invocation still needs position-indexed storage (Option A) underneath.

**Option C — Convergence-Based Looping**
Loop a sub-chain until a condition is met rather than a fixed count:
```
"writer -> reviewer -> writer@until(approved, max=5)"
```
Executor runs the `[writer -> reviewer]` block repeatedly, checking the condition after each full pass. Combines repetition with quality control.
- Pros: Self-regulating — stops as soon as quality is sufficient, not after a fixed N.
- Cons: Requires condition evaluation machinery (shares infrastructure with conditional branching above); harder to predict runtime.

**Recommendation:** Option A is a pure bug-fix with minimal risk and is a prerequisite for both B and C.

---

#### Weighted Concurrent Execution Options

Currently comma-separated agents in a flow step all run identically with no priority ordering.

**Option A — Priority-Ordered Result Collection**
Add weight annotation to the flow syntax. Results are still fetched in parallel but added to the conversation in weight order, so downstream agents see the highest-priority output first:
```
"agent1 -> critical_agent:3, helper_agent:1 -> agent4"
```
```python
# Parse weights, sort descending, collect results in weight order
weighted_agents = [("critical_agent", 3), ("helper_agent", 1)]
for name, _ in weighted_agents:  # sorted by weight desc
    self.conversation.add(name, futures[name].result())
```
- Pros: Simple, no execution change — only affects conversation ordering.
- Cons: Doesn't actually prioritize thread scheduling.

**Option B — Timeout-Based Weighting**
Low-weight agents get shorter timeouts. Results that don't arrive in time are skipped:
```python
for name, weight in weighted_agents:
    timeout = weight * base_timeout  # weight=3 → 3x the base timeout
    try:
        result = future.result(timeout=timeout)
    except TimeoutError:
        result = None  # skip slow low-priority agent
```
- Pros: Protects the workflow from slow low-priority agents holding things up.
- Cons: Permanently discards results that would have been useful given more time.

**Option C — Resource Allocation Weighting**
Weights control how many internal workers each agent can use when the agent itself does parallel work:
```python
total_weight = sum(w for _, w in weighted_agents)
for name, weight in weighted_agents:
    worker_share = max(1, int(available_workers * weight / total_weight))
    self.agents[name].run(task=..., max_workers=worker_share)
```
- Pros: Useful when agents internally parallelize (batch LLM calls, tool use).
- Cons: Requires agents to accept and respect a `max_workers` hint.

**Option D — Quorum / Early Exit** *(recommended)*
Once agents with enough combined weight have finished, skip waiting for the rest:
```
"agent1 -> critical:5, backup1:2, backup2:2 -> agent4"
# quorum=5: once critical finishes (weight 5 ≥ quorum), proceed immediately
```
```python
accumulated = 0
for future in as_completed(futures):
    name, weight = futures[future]
    results[name] = future.result()
    accumulated += weight
    if accumulated >= quorum:
        [f.cancel() for f in futures if not f.done()]
        break
```
- Pros: Biggest practical impact — a slow low-priority agent no longer blocks the step.
- Cons: Cancelled agents produce no output; downstream agents must handle missing inputs.

| Option | Scheduling impact | Result impact | Risk |
|--------|------------------|---------------|------|
| A. Priority ordering | None | Conversation order only | Very low |
| B. Timeout weighting | None | May discard results | Medium |
| C. Resource allocation | Thread-level | Agent-internal parallelism | Medium |
| D. Quorum / early exit | Cancels slow futures | Some results missing | Medium |

**Recommendation:** Ship Option A (conversation ordering) first — zero risk. Then Option D (quorum) for the main latency benefit.

---

## GraphWorkflow

**File:** `swarms/structs/graph_workflow.py`

### Execution & Routing

| # | Improvement | Description |
|---|-------------|-------------|
| 1 | **Conditional / dynamic routing** | All edges are static — there is no way to route to a different node based on an agent's output at runtime. Adding conditional edges with a predicate `(output) -> bool` would unlock decision trees, retry loops, and branching workflows without rebuilding the graph. |
| 2 | **True async execution** | `arun` is fake async — it wraps synchronous `run` in `run_in_executor`. Since agents make network calls (LLM APIs), true `asyncio`-native execution with `agent.arun` calls and `asyncio.gather` per layer would massively improve throughput and eliminate thread overhead. |
| 3 | **`max_loops` returns after first loop** | There is an explicit comment at line 1897: *"For now, we still return after the first loop — this maintains backward compatibility."* The `while loop` runs but `return` exits on the first iteration. The multi-loop logic should accumulate results across loops or be removed entirely. |
| 4 | **Per-node timeout and retry** | No timeout or retry mechanism exists per agent. A slow or failing agent blocks an entire layer indefinitely. Exposing `timeout_seconds` and `max_retries` on `Edge` or `Node` metadata would add fault tolerance without changing the execution model. |
| 5 | **Persistent thread pool across layers** | A new `ThreadPoolExecutor` is created and torn down for every layer (line 1788). A single executor scoped to the entire `run` call would avoid repeated thread creation overhead, especially in multi-loop or many-layer scenarios. |

### Data Flow

| # | Improvement | Description |
|---|-------------|-------------|
| 6 | **Edge output transform functions** | All predecessor outputs are concatenated into a single prompt with no filtering. Adding an optional transform `(source_output: str) -> str` on `Edge` would let users summarize, restructure, or selectively pass data between nodes, enabling cleaner information handoff. |
| 7 | **Output aggregation strategy** | `run` returns a flat `Dict[str, Any]` keyed by node ID with no built-in aggregation for the final answer. Adding an `output_mode: Literal["all", "end_points", "last_layer"]` parameter would make consuming results much cleaner without requiring post-processing by the caller. |
| 8 | **Prompt builder customization** | `_build_prompt` uses a hardcoded template for all nodes. Exposing a `prompt_template: Optional[Callable]` on `GraphWorkflow` or per-node would let users control exactly how predecessor context is formatted for each agent instead of the one-size-fits-all format. |

### Reliability & Error Handling

| # | Improvement | Description |
|---|-------------|-------------|
| 9 | **Fallback nodes on agent failure** | When an agent fails, the error string is passed downstream as context with no recovery path. Supporting a `fallback_node_id` on `Node` or a catch-edge pattern would let the graph route around failures instead of poisoning downstream agents with error text. |
| 10 | **Streaming / intermediate output callbacks** | There is no way to observe results as they complete within a layer. A `on_node_complete(node_id, output)` callback hook would enable real-time UIs, logging pipelines, and early-exit patterns without polling the final result dict. |
| 11 | **Checkpoint and resume** | Long-running workflows have no way to persist completed node outputs and resume from a mid-graph failure. A `checkpoint_dir` parameter that saves `prev_outputs` to disk after each layer would make the system production-grade. |
| 12 | **Explicit graph validation API** | There is no `validate()` method users can call before running. Surfacing checks for disconnected nodes, missing entry/end points, cycle detection, and unreachable nodes as a standalone method would improve DX and surface misconfiguration before any agents are invoked. |

### Architecture & Extensibility

| # | Improvement | Description |
|---|-------------|-------------|
| 13 | **Expand `NodeType` beyond agents** | `NodeType` is an enum with a single value (`AGENT`). Supporting **tool nodes** (pure functions), **router nodes** (conditional dispatch), and **human-in-the-loop nodes** (pause + resume) would make GraphWorkflow a general-purpose workflow engine rather than an agent-only orchestrator. |
| 14 | **Workflow serialization / deserialization** | There is no `to_json` / `from_json` or equivalent. Users cannot save a workflow spec, version it in git, or reconstruct it from config. A `to_spec()` capturing agent names and edge structure would unlock workflow persistence and sharing without requiring agent objects to be serializable. |

### Backend & Infrastructure

| # | Improvement | Description |
|---|-------------|-------------|
| 15 | **RustworkX `simple_cycles` falls back to NetworkX** | The rustworkx backend's `simple_cycles` silently converts the graph to NetworkX for cycle detection (lines 499–511), defeating the purpose of the backend abstraction. Either implement native DFS cycle detection in rustworkx or clearly document the fallback so users know the performance guarantee does not hold for cycle detection. |

---

## Cross-Cutting Improvements

Improvements that apply to all multi-agent architectures.

| # | Area | Improvement | Description |
|---|------|-------------|-------------|
| 1 | **Observability** | OpenTelemetry tracing | Add tracing spans per agent call for latency profiling across all swarm types. |
| 2 | **Token Budgeting** | Per-agent token limits | Set per-agent token limits to prevent one agent from consuming the entire context window. |
| 3 | **Caching** | LLM response cache | Cache identical LLM calls (same prompt = same result) across agents and runs. |
| 4 | **Warm-up** | Pre-initialize connections | Pre-initialize agent connections/models before workflow starts to avoid cold-start latency on the first agent. |
| 5 | **Serialization** | Structured message objects | Use structured message objects instead of string concatenation for inter-agent communication. |

---

## GitHub Issues Tracking

| Issue | Title | Architecture |
|-------|-------|-------------|
| [#1458](https://github.com/kyegomez/swarms/issues/1458) | O(1) agent lookup + parallel execution by default | HierarchicalSwarm |
| [#1460](https://github.com/kyegomez/swarms/issues/1460) | Cache `Conversation.get_str()` with invalidation | AgentRearrange / All |
| [#1461](https://github.com/kyegomez/swarms/issues/1461) | Parse flow once at init, not per-run | AgentRearrange |
| [#1462](https://github.com/kyegomez/swarms/issues/1462) | Drop redundant `any_to_str()` calls | AgentRearrange |
| [#1463](https://github.com/kyegomez/swarms/issues/1463) | Pipeline parallelism + fix `batch_run` ThreadPoolExecutor | AgentRearrange |
| [#1464](https://github.com/kyegomez/swarms/issues/1464) | `_run()` silently swallows exceptions | AgentRearrange |
| [#1465](https://github.com/kyegomez/swarms/issues/1465) | `run()` double-logs errors on failure | AgentRearrange |
| [#1466](https://github.com/kyegomez/swarms/issues/1466) | Support agent repetition in flow | AgentRearrange |
