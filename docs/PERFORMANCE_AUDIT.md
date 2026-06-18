# Performance Audit — `swarms/structs/`

Scope: 61 files, ~46,800 LOC. Focused on hot paths in `Agent`, `Conversation`, and the major multi-agent orchestrators. Findings are grouped by **root cause** so the same fix often resolves issues across files.

---

## TL;DR — Top 10 Wins (by ROI)

| # | Issue | Files | Impact | Effort |
|---|---|---|---|---|
| 1 | Full conversation rebuilt to string on every loop / agent hop | `agent.py`, `conversation.py`, `agent_rearrange.py`, `mixture_of_agents.py` | **HIGH** | M |
| 2 | Token counts recomputed instead of cached / incremental | `agent.py`, `conversation.py`, `transforms.py` | **HIGH** | S |
| 3 | Autosave + MEMORY.md re-serialised on every message/loop | `agent.py:1611-1614`, `conversation.py:419-446, 812-893` | **HIGH** | S |
| 4 | Topological sort & graph compilation re-run on every `run()` | `graph_workflow.py:232-240, 409-492, 1084-1097` | **HIGH** | M |
| 5 | Tool-summary path makes a 2nd LLM call per tool | `agent.py:6263-6280` | **HIGH** | S |
| 6 | Swarm/agents re-instantiated on every `run()` | `swarm_router.py:256-258`, `heavy_swarm.py:210` | **HIGH** | M |
| 7 | Eager top-level imports load all 46k LOC + heavy SDKs | `swarms/structs/__init__.py`, `swarm_router.py:17-44`, `agent.py:1-130` | **MED-HIGH** | S |
| 8 | ThreadPoolExecutor created per call, never reused | `agent.py:3566`, `concurrent_workflow.py:335`, `base_structure.py:374` | **MED** | S |
| 9 | Thread-unsafe shared dicts in concurrent fan-out | `concurrent_workflow.py:243-276` | **HIGH** (correctness) | S |
| 10 | O(n²) middle-out message compression | `transforms.py:273-328` | **HIGH** in long sessions | M |

Hitting just **#1–#5** should cut typical autonomous-agent loop latency by **~30–50%** and large multi-agent workflow time by **~30–70%**.

---

## 1. Conversation: the single biggest hot path

`Conversation` sits underneath every agent and every orchestrator. Almost every other perf issue traces back to it.

### 1.1 `return_history_as_string()` is rebuilt O(n) times per loop
- `agent.py:1668, 2345-2346, 5346` each call it independently per loop iteration; with N loops on a conversation of length M, that's **O(N·M)** string work.
- `agent_rearrange.py:559, 644` calls `conversation.get_str()` per agent hop — sequential pipelines grow quadratically.
- `mixture_of_agents.py:171-182` re-passes `full_context = conversation.get_str()` to every worker in every layer. Layer 2 with 100 workers on a 20 KB history passes ~2 MB of context.

**Fix:** memoise the serialised string on `Conversation` and invalidate it from `add()` / `add_multiple()` / `compact()`. Compute the delta once per turn, not per reader. Internally store both `messages: list[dict]` and a lazy `_str_cache: Optional[str]`.

### 1.2 Token counting is unbatched and re-run on every add
- `conversation.py:485-490` tokenises every message synchronously on `add()`.
- `transforms.py:273-328` (`_middle_out_compress_tokens`) re-counts the **entire** message list on every loop iteration of the trim → **O(n²)** tokenisation.
- `conversation.py:1488-1490, 1130-1141` binary-search loops invoke `count_tokens()` `O(log n)` times per call with no memo.

**Fix:**
1. Store `token_count` alongside each message on `add()`. Maintain a running `_total_tokens` counter. All callers read this instead of re-tokenising.
2. For `_middle_out_compress_tokens`, subtract token counts incrementally as messages are dropped from the middle.
3. Wrap `count_tokens(model, text)` in an `lru_cache` keyed on `(model, hash(text))` for the unavoidable repeated calls.

### 1.3 `MEMORY.md` is written synchronously on every `add()`
- `conversation.py:419-446` opens/writes/closes `MEMORY.md` per message. In an autonomous loop with tool calls, that's dozens of fsync-cost writes per turn.
- `conversation.py:812-893` (`save_as_json`, `save_as_yaml`) rewrites the entire history on every autosave call.
- `agent.py:1611-1614, 1812-1813, 1824` calls full `to_dict()` + JSON save on **every loop iteration** when `autosave=True`.

**Fix:**
1. **Append-only JSONL** for autosave (`save_as_jsonl(message)` appends one line). Keep a periodic checkpoint that consolidates into the legacy JSON layout.
2. Buffer `MEMORY.md` writes with a small queue flushed every N messages or every K ms — fsync amortises across the batch.
3. Throttle `_autosave_config_step` to every M loops (M=5–10) or only when state actually changes.

### 1.4 `add_multiple()` thread-fans out to single `add()` calls
- `conversation.py:606-627` uses a 25% CPU ThreadPoolExecutor to call `add()` per message. Each thread contends on the same write lock and triggers its own `_append_to_memory_md()` and autosave path.

**Fix:** acquire the lock once, append all messages in-memory, then do one batched MEMORY.md append + one autosave. The thread fan-out is anti-helpful for CPU-bound list appends.

### 1.5 Other smaller wins inside `Conversation`
- `conversation.py:431` `datetime.now()` syscall per message — pass the timestamp through from `add()` once.
- `conversation.py:1058-1108` `truncate_memory_with_tokenizer` iterates **forward**; iterate from the tail and break early (recency-priority).
- `conversation.py:1307-1318` `return_all_except_first_string` duplicates the worker logic — implement as a slice of the cached string.

---

## 2. `Agent` — the hot path inside the hot path

### 2.1 Tool-call summary causes a 2nd LLM round-trip per tool
- `agent.py:6263-6280` instantiates an ephemeral LLM and re-prompts to summarise tool output when `tool_call_summary=True`. This **doubles** the LLM cost of every tool call.

**Fix:** fold the summary instruction into the main system prompt, or stream-process the result inline. The 2nd call is rarely worth its cost.

### 2.2 `reliability_check` hits `litellm.model_list` twice
- `agent.py:3300-3316` calls `supports_function_calling()` and `get_max_tokens()` redundantly. Both load the full litellm registry.

**Fix:** call once, store on `self`. Make litellm a lazy import (`agent.py:1-130` currently eager-imports it even when irrelevant).

### 2.3 Linear scans of `autonomous_subtasks`
- `agent.py:3039, 5601, 5705, 5729, 5760, 5785` repeatedly scan the list for status checks. A `subtask_by_id` dict exists in places but isn't consistently used.

**Fix:** maintain `_status_counts: dict[str,int]` and update it on every transition. Status queries become O(1).

### 2.4 Streaming hot loop builds 15+ f-strings per token
- `agent.py:4299, 4313-4349` formats per-token output with many `getattr` lookups and f-strings. For a 1000-token response that's ~15k temporary strings.

**Fix:** precompute the per-message template prefix once, append raw token text in the loop, format the wrapping panel once at the end.

### 2.5 Per-token JSON re-parsing for tool args
- `agent.py:1750-1754, 2372-2376, 2398-2401` calls `json.loads(tool_call["function"]["arguments"])` multiple times across visualisation and execution paths.

**Fix:** parse once at the call-site, pass the dict.

### 2.6 Logging hot paths build expensive payloads unconditionally
- `agent.py:1832, 3201-3202` build `traceback.format_exc()` before checking log level; `agent.py:1959, 3578-3601` build full `to_dict()` for verbose logs.

**Fix:** guard with `logger.isEnabledFor(...)` or switch to `logger.exception(...)`. For verbose summaries, log a 5-field dict (loop, model, msg_count, tokens, status), not the whole state.

### 2.7 ThreadPoolExecutor leaks
- `agent.py:3566-3569, 5274, 6323` and several places in workflows create a new executor per call instead of reusing `self.executor`. Pool creation is cheap but not free, and these calls happen in tight loops.

**Fix:** lazy-create one `self._executor` per instance, reuse, cleanly shut down on `__del__`/`close()`.

### 2.8 Agent-name sanitisation (`agent.py:758-774`)
- 11 chained `.replace(...)` calls (plus a duplicate `.replace("--","-")` on lines 772–773). One regex pass would be both faster and clearer.

---

## 3. Orchestrators

### 3.1 `concurrent_workflow.py`
- **Race condition**: `agent_statuses` dict mutated from worker callbacks (`243-249, 265-276`) with no lock. Lost-update bugs in the dashboard.
- **Oversized pool**: `0.95 * cpu_count` workers (`253, 401`). LLM calls are I/O-bound but 95% CPU saturation leaves no headroom for other work in the same process.
- **Executor churn**: new TPE per `run()` (`335-337, 403-405`). Keep one persistent executor on the workflow.

**Fix:** protect `agent_statuses` with a `Lock` (or move to `queue.Queue`); expose `max_workers` and default to a saner I/O multiplier (e.g. `min(32, 4 * cpu_count())` — the Python stdlib default).

### 3.2 `sequential_workflow.py`
- **Drift-detection agent eagerly constructed** (`134-146`) even when `drift_detection=False` → 100 ms+ wasted init.
- **Unbounded retries on drift miss** (`231-257`) — no max attempts, runs the whole pipeline again each time.

**Fix:** lazy-init drift agent on first miss; add `max_drift_retries` with exponential backoff.

### 3.3 `agent_rearrange.py`
- `conversation.get_str()` on every hop (`559, 644`) — see §1.1.
- System-prompt mutate/restore (`633-651`) recomputes awareness strings on every reuse of the same agent.
- `copy.deepcopy(agent)` with silent fallback (`895-901`) — when deepcopy fails (locks, file handles), the fallback **shares the original**, silently breaking batch isolation. Use a config-based shallow clone instead.
- Flow string is `.split("->")`ed on every run (`707`) — cache as `_flow_tasks` on `set_custom_flow()`.

### 3.4 `graph_workflow.py`
- **Topological sort re-run every `compile()`** (`232-240, 409-492`). For repeated runs on the same DAG this is pure waste. Cache `_sorted_layers`, invalidate only on structural mutation.
- **Per-node invalidation** (`1084-1097`): adding 100 nodes triggers 100 recompiles. Add a `_batch_mutate()` context manager that defers invalidation to exit.
- **Reversed graph rebuilt** on each ancestor/descendant query (`221-231, 388-407`) — cache `_reversed_graph` with the same invalidation guard.
- **Edge list has no dedup** (`1145, 1209, 1281, 1356`) — duplicate edges silently inflate fan-in cost.

### 3.5 `hiearchical_swarm.py`
- Dashboard rebuilds **all** Rich panels on every status tick (`282-320, 450-456`) — only the changed panel needs to refresh.

### 3.6 `mixture_of_agents.py`
- See §1.1. Passing `full_context` to every worker in every layer is the dominant cost. Pass the **task** + the **previous-layer aggregated output**, not the full transcript.

### 3.7 `heavy_swarm.py`
- `time.sleep()` calls in `reliability_check` (`333, 345, 357, 369`) add ~2 s to every instantiation for cosmetic reasons. Remove.
- `create_agents()` re-runs on every loop (`210`) — build once in `__init__`.

### 3.8 `swarm_router.py`
- `_swarm_cache` exists (`258`) but isn't used — each `run()` re-instantiates the swarm. Cache by `(swarm_type, agent_ids, config_hash)`.
- All swarm types imported at module top (`17-44`) — a 500 ms+ cold-start tax to use just one. Switch to a lazy factory: `_FACTORIES = {"SequentialWorkflow": lambda: import_module(...).SequentialWorkflow, ...}`.

### 3.9 `multi_agent_exec.py`
- `asyncio.get_event_loop()` (`227, 232`) is deprecated/broken in thread contexts and Python 3.10+. Use `asyncio.run()` or `asyncio.new_event_loop()`.
- No per-task timeout in the concurrent-fanout path (`152-160`) — one hanging agent blocks the whole batch.

---

## 4. Cross-cutting issues

### 4.1 Package import cost
`swarms/structs/__init__.py` eagerly imports all 61 modules (`AOP` alone is 2,948 LOC). A user who only needs `Agent` still pays for `GraphWorkflow`, `HeavySwarm`, `AOP`, etc., plus their transitive deps.

**Fix:** use PEP 562 `__getattr__` lazy re-exports:

```python
# swarms/structs/__init__.py
_LAZY = {
    "GraphWorkflow": ("swarms.structs.graph_workflow", "GraphWorkflow"),
    "HeavySwarm":    ("swarms.structs.heavy_swarm",    "HeavySwarm"),
    # ...
}
def __getattr__(name):
    if name in _LAZY:
        mod, attr = _LAZY[name]
        return getattr(importlib.import_module(mod), attr)
    raise AttributeError(name)
```

### 4.2 Repeated regex compilation
`cron_job.py:121`, agent-name sanitisation. Hoist to module-level `_COMPILED = re.compile(...)`.

### 4.3 Serialisation round-trips
- `base_structure.py:333-356` does `json.dumps(json.dumps(data))` before gzip — double encoding, ~2× the bytes.
- `base_structure.py:486-510` `to_dict()` runs a `json.dumps` *test* per attribute to decide if it's serialisable. Use `isinstance` against a tuple of primitive types first; fall back to `json.dumps` only on the ambiguous cases.
- `safe_loading.py:43-84` has unbounded recursion — add a `max_depth` argument.

### 4.4 AOP task queue is O(n) per enqueue
- `aop.py:213-220` uses `deque.insert` with a linear priority search. With 10k queued tasks this is O(n²) ingestion.

**Fix:** use `heapq` (priority queue) or `bisect.insort` on a sorted list, keyed on `(-priority, seq)` for stable ordering.

### 4.5 Linear `get_agent_by_name`
- `base_swarm.py:236-240` scans the list every call despite `agents_dict` already being built (`183-185`). Just use the dict.

### 4.6 Validation costs on init
- `base_swarm.py:149-150` validates all agents and callbacks on every `__init__`. For frequently-rebuilt swarms (see §3.8) this compounds.

**Fix:** validate lazily on first `run()`; skip for known-clean clones.

---

## 5. Suggested implementation order

1. **`Conversation` overhaul** (§1) — single biggest win, touches every other module. Add `_str_cache`, per-message token counts, append-only JSONL autosave, batched MEMORY.md writes.
2. **Agent autosave / autoreliability cleanup** (§2.1, §2.2, §2.6, §2.7) — small surface area, large impact.
3. **`SwarmRouter` lazy imports + swarm caching** (§3.8) and **package `__init__` PEP 562** (§4.1) — fixes import latency and re-instantiation cost in one stroke.
4. **`graph_workflow` compilation cache + batched mutation** (§3.4) — separable, well-bounded refactor.
5. **`agent_rearrange` deepcopy fix** (§3.3) — correctness bug masquerading as a perf issue. High priority.
6. **`transforms.py` incremental token tracking** (§1.2) — unlocks middle-out compression at scale.
7. **Race fix in `concurrent_workflow`** (§3.1) — correctness fix, do it before any of the perf-only work above ships.

---

## 6. Suggested benchmarks to add

To avoid regressing these gains, add micro-benchmarks under `tests/perf/`:

- `bench_conversation_add.py` — N=10_000 adds with/without token counting, with/without MEMORY.md.
- `bench_agent_loop.py` — `max_loops=10` on a trivial tool, measure wall-clock and LLM-call count (mock model).
- `bench_graph_workflow.py` — 100-node DAG, 1000 runs, measure compilation cost.
- `bench_swarm_router.py` — 1000 `run()` calls with the same agents, verify swarm reuse.
- `bench_concurrent_workflow.py` — 100 agents × 1 task with mock LLM, assert no race conditions via assertions on output ordering / counts.

A simple `pytest-benchmark` setup is enough; gate CI on a regression threshold of ~10%.
