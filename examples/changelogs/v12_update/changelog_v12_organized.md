# Changelog v12 — April 18 → May 2, 2026

> Apache 2.0 · [github.com/kyegomez/swarms](https://github.com/kyegomez/swarms)
>
> **Contributors:** Kye Gomez · Steve-Dusty · adichaudhary · MycCellium420

---

## Introduction

Over a focused two-week sprint from April 18 to May 2, 2026, the swarms framework landed 35 commits from four contributors, broken down into 8 new features, 7 improvements, and 4 bug fixes, alongside 2 new test suites and 4 docs updates. Despite the volume of new functionality, the release is a net reduction of roughly 1 700 lines, as the senator-assembly module and the unused uvloop and winloop execution paths were retired in favour of a leaner core.

The headline additions all centre on agent memory and observability: persistent memory that survives process restarts by default, an automatic context compressor that summarises history once token usage crosses 90 %, a safe grep tool for the autonomous loop, and per-node completion plus token-streaming callbacks for GraphWorkflow. Conversation compaction now writes timestamped archive snapshots before rewriting active memory, and interactive mode picks up a Rich loading spinner and graceful keyboard-interrupt handling.

---

## Features

### Persistent Memory (`persistent_memory`)

**2026-05-02** · `e48100a9` · Kye Gomez

Added `persistent_memory: bool = True` to `Agent`. When `True` (default) the agent reads and writes a `MEMORY.md` file under `$WORKSPACE_DIR/agents/{agent_name}/MEMORY.md`, loading prior conversation history as a system preamble on every startup so state survives process restarts. Set to `False` for a fully stateless agent — no disk reads or writes, every run starts from a blank slate.

```python
# Stateless — no cross-session carry-over
agent = Agent(
    agent_name="MyAgent",
    model_name="gpt-4.1",
    persistent_memory=False,
)
```

---

### Grep Tool for Autonomous Loop

**2026-05-02** · `75e12876` · Kye Gomez

Added a `grep` tool to the autonomous loop tool set. The agent can now search for patterns in files without falling back to `run_bash`. Arguments are passed as an argv list (no `shell=True`) to prevent injection. Output is capped at 64 KB.

Parameters: `pattern` (required), `path`, `recursive`, `case_insensitive`, `include_line_numbers`, `file_pattern`, `context_lines`.

```python
agent = Agent(agent_name="Coder", model_name="gpt-4.1", max_loops="auto")
agent.run("Find all TODO comments in the src/ directory")
```

---

### Persistent Conversation Memory on Disk

**2026-04-20** · `9205bd16` · Kye Gomez

`Conversation` now persists every message to a `MEMORY.md` file and reloads it as a system preamble on the next instantiation. The file path is keyed on `agent_name` (not the ephemeral `id`) so memory is stable across process restarts.

---

### Context Compressor — Auto-Summarise at Token Threshold

**2026-04-20** · `9205bd16` · Kye Gomez

Added `ContextCompressor`, activated at the top of every loop iteration when token usage crosses 90 % of `context_length`. It summarises and rewrites `MEMORY.md` in place to keep the context window healthy during long sessions. Controlled via `context_compression: bool = True` on `Agent`.

---

### Conversation Compact with Archive Snapshots

**2026-04-20** · `9205bd16` · Kye Gomez

`Conversation.compact()` creates a timestamped archive snapshot before rewriting the active `MEMORY.md`, so the full history is recoverable even after compression.

---

### Interactive Mode — Graceful Exit on Keyboard Interrupt

**2026-04-20** · `9205bd16` · Kye Gomez

`Ctrl+C` in interactive mode now exits cleanly instead of raising an unhandled `KeyboardInterrupt`.

---

### Interactive Mode — Rich Loading Spinner

**2026-04-20** · `9205bd16` · Kye Gomez

Added a Rich animated spinner displayed while the agent processes a task in interactive mode.

---

### GraphWorkflow — `on_node_complete` and `streaming_callback`

**2026-05-01 (merged)** · `f09e753f` · Steve-Dusty

`GraphWorkflow.run()` now accepts an `on_node_complete` callback that fires after each node completes, and a `streaming_callback` that forwards individual response tokens in real time.

```python
def on_node_complete(node_name: str, result: str) -> None:
    print(f"[{node_name}] done")

workflow.run(task="Analyse this dataset", on_node_complete=on_node_complete)
```

---

## Improvements

### Autonomous Loop — Execution Prompt Written Once Per Subtask

**2026-05-02** · `75e12876` · Kye Gomez

The execution prompt was added to `short_memory` on every inner-loop iteration, causing the model to see duplicate context and treat subsequent iterations as repeated work. Moved the `short_memory.add` call to outside the inner while loop so it fires exactly once per subtask.

---

### Autonomous Loop — Exclude `think` Tool When Native Thinking Is Enabled

**2026-05-02** · `75e12876` · Kye Gomez

When `thinking_tokens` is set the model already reasons via extended thinking. The `think` tool created unnecessary extra round-trips. It is now filtered from `planning_tools` when `thinking_tokens is not None`.

---

### `arun_stream` — Use `get_running_loop` and Propagate Exceptions

**2026-05-02** · `75e12876` · Kye Gomez

`arun_stream()` used the deprecated `asyncio.get_event_loop()` and silently swallowed exceptions from the background thread. Fixed to use `asyncio.get_running_loop()` and propagate exceptions to the async consumer.

---

### Performance — Guard `any_to_str()` with `isinstance` Check

**2026-04-20** · `4824fe82` · adichaudhary

Added an `isinstance` short-circuit before calling `any_to_str()` in `AgentRearrange`, avoiding an unnecessary conversion when the value is already a string.

---

### Remove `uvloop` / `winloop` Dependencies

**2026-04-19** · `9879c4f1` · Kye Gomez / Steve-Dusty / MycCellium420

Removed `uvloop` and `winloop` from the dependency list and deleted the dead execution functions that depended on them. Reduces install size and eliminates a platform-specific dependency that was never activated in production.

---

### Timestamps in Conversation History String

**2026-04-20** · `9205bd16` · Kye Gomez

`Conversation.return_history_as_string()` now includes ISO timestamps on each message entry, making exported history easier to audit and correlate with logs.

---

### Senator Assembly Module Removed

**2026-05-02** · `75e12876` · Kye Gomez

Deleted the deprecated `swarms/sims/senator_assembly.py` (3 483 lines) and its associated example and `__init__` re-export.

---

## Bug Fixes

### Multi-Provider — Non-OpenAI Providers Producing No Output

**2026-04-23** · `9d250edf` · Steve-Dusty

Fixed a regression where providers other than OpenAI (Anthropic, Google, etc.) returned an empty string instead of their response content when `reasoning_effort` was set.

---

### Streaming — Thinking Panel Corruption Inside Rich Live Context

**2026-05-02** · `75e12876` · Kye Gomez

`console.print()` called from inside a generator consumed by a `rich.Live` block caused terminal corruption (overlapping panels, cursor misplacement). Fixed by pre-draining thinking chunks and printing the thinking panel *before* the `Live` context opens.

---

### Autonomous Loop — `_generate_final_summary` Dropped `streaming_callback`

**2026-05-02** · `75e12876` · Kye Gomez

`_generate_final_summary` was not forwarding `streaming_callback` to `call_llm`, so the summary phase streamed silently even when a callback was registered. Fixed by threading `streaming_callback` through the method signature.

---

### License Metadata — Corrected MIT → Apache 2.0

**2026-05-02** · `75e12876` · Kye Gomez

`pyproject.toml` declared `license = "MIT"` and the corresponding OSI classifier, conflicting with the actual Apache 2.0 `LICENSE` file. Both fields corrected to `Apache-2.0`.

---

## Tests

### Conversation — MEMORY.md Persistence, Compact, and Timestamp Coverage

**2026-04-23** · `fbc8a2bd` · Steve-Dusty

Added a dedicated test suite covering MEMORY.md round-trip persistence, compact-with-archive behaviour, and timestamp formatting in `return_history_as_string`.

---

### Agent Streaming and Autonomous Loop — Real-LLM Test Suite

**2026-05-02** · `75e12876` · Kye Gomez

Added `test_agent_streaming_and_loop.py` — 71 tests across 19 classes using real `Agent` and `LiteLLM` instances (no mocked LLM). Covers streaming pipeline correctness, `arun_stream` async behaviour, thinking-panel rendering, autonomous loop harness fixes, and `_generate_final_summary` callback threading.

---

## Docs

### Agent Memory Guide

**2026-04-20** · `9205bd16` · Kye Gomez

New guide at `docs/swarms/agents/agent_memory.md` covering the full MEMORY.md flow, `ContextCompressor` activation, and compact-with-archive behaviour.

---

### Agent Docs — Persistent Memory and Context Compression Examples

**2026-05-02** · `e48100a9` · Kye Gomez

Added a "Memory Persistence and Context Compression" section to `docs/swarms/structs/agent.md` with annotated code examples for `persistent_memory`, `context_compression`, and combined usage patterns.

---

### GraphWorkflow Streaming Callback Example

**2026-05-01** · `80824067` · Steve-Dusty

Added a complete runnable example demonstrating `on_node_complete` and `streaming_callback` usage with `GraphWorkflow`.

---

### README Updates

**2026-04-30** · Kye Gomez

Multiple README passes: updated import paths for `SwarmRouter`, `AutoSwarmBuilder`, and `AOP` to use the top-level `swarms` package; updated model name references and integration examples.

---

## Conclusion

v12 is fundamentally a memory and reliability release. The combination of persistent memory, the context compressor, conversation compaction with archive snapshots, and on-disk message history transforms the agent from a stateless request handler into a long-lived, restart-safe entity capable of running indefinitely without losing context, while the new grep tool meaningfully extends what the autonomous loop can do on real codebases.

The supporting work is just as important: 7 improvements tighten the autonomous loop and shed dead dependencies, 4 bug fixes resolve real production issues across non-OpenAI providers, streaming, and the final-summary phase, and 2 new test suites — including a 71-test real-LLM harness — back the new paths with end-to-end coverage rather than mocks. Across 35 commits and a net reduction of around 1 700 lines, the framework comes out of this sprint smaller, sharper, and more capable than it went in.

---

## Stats

| Metric | Value |
| --- | --- |
| Period | 2026-04-18 → 2026-05-02 |
| Total commits | 35 |
| Contributors | Kye Gomez, Steve-Dusty, adichaudhary, MycCellium420 |
| Lines added | ~2 500 |
| Lines removed | ~4 200 |
| Net | −1 700 (dead-code removal) |
