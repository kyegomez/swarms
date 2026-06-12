# swarms v12 — A Memory and Reliability Release

> Apache 2.0 · [github.com/kyegomez/swarms](https://github.com/kyegomez/swarms)
>
> **Contributors:** Kye Gomez · Steve-Dusty · adichaudhary · MycCellium420

---

## Introduction

Over a focused two-week sprint from April 18 to May 2, 2026, the swarms framework landed 35 commits from four contributors, broken down into 8 new features, 7 improvements, and 4 bug fixes, alongside 2 new test suites and 4 docs updates. Despite the volume of new functionality, the release is a net reduction of roughly 1 700 lines, as the senator-assembly module and the unused uvloop and winloop execution paths were retired in favour of a leaner core.

The headline additions all centre on agent memory and observability: persistent memory that survives process restarts by default, an automatic context compressor that summarises history once token usage crosses 90 %, a safe grep tool for the autonomous loop, and per-node completion plus token-streaming callbacks for GraphWorkflow. Conversation compaction now writes timestamped archive snapshots before rewriting active memory, and interactive mode picks up a Rich loading spinner and graceful keyboard-interrupt handling.

---

## Features

### Persistent Memory

Introducing persistent memory. Every agent now keeps its own long-term notebook on disk, so when you start it again tomorrow, next week, or after a crash, it remembers everything from before — there is nothing extra to set up, it just works the moment you create the agent. Persistent memory helps with long-running projects, ongoing conversations with a single agent, and any workflow where you do not want to re-explain context every time the process restarts. If you want a clean-slate agent for a one-off task, you can turn it off with a single switch.

### Grep Tool for the Autonomous Loop

Introducing the grep tool. When an autonomous agent needs to find something inside a codebase or a folder of text files, it can now search directly with a built-in tool instead of falling back to running shell commands. The grep tool helps with code exploration, finding TODOs, locating where a function is used, and any other "search the files for X" task — and because the search arguments are never fed into a shell, malicious input cannot turn into a command injection. The result is automatically capped in size so a huge match cannot flood the agent's working memory.

### Persistent Conversation Memory on Disk

Introducing on-disk conversation memory. The Conversation primitive — the part of the framework that holds an agent's chat history — now writes every message to a file as it happens and reloads that file the next time the same agent starts up. This helps with crash recovery, long-running tasks that span days, and any case where you want the agent's full message history to outlive a single Python process. The file is named after the agent rather than the run, so memory follows the agent identity instead of vanishing when the process exits.

### Context Compressor

Introducing the context compressor. Long-running agents used to eventually fill up their context window — the memory limit a model can hold in a single turn — and crash mid-task. The compressor helps with exactly this: it watches the conversation as it grows, and once it gets close to the limit it automatically summarises the older parts so the agent can keep going indefinitely without ever hitting the wall. You set a context length, and the framework keeps you safely under it.

### Conversation Compact with Archive Snapshots

Introducing compaction with archive snapshots. Whenever the framework summarises a long conversation to save space, it now first saves a complete copy of the original, untouched conversation with a timestamp on it. This helps with debugging, auditing, and post-mortems — even after the live memory has been condensed, you can always go back and see exactly what was said. Nothing is ever truly thrown away.

### Graceful Exit on Keyboard Interrupt

Introducing graceful exit. When you press Ctrl+C to stop an agent in interactive mode, it now shuts down cleanly and returns you to the prompt instead of crashing with a stack trace. This helps with everyday use, where you frequently want to interrupt a long generation or back out of a session without seeing an unhandled error each time. Small change, big quality-of-life win.

### Rich Loading Spinner in Interactive Mode

Introducing a loading spinner. While the agent is thinking and you are sitting in interactive mode waiting for a response, you now see an animated spinner instead of a frozen-looking screen. This helps with the constant "is this thing actually running?" feeling that came with longer tasks — you immediately know the agent is alive and working, even on requests that take a while.

### GraphWorkflow Callbacks

Introducing GraphWorkflow callbacks. When you run a graph of agents, you can now hook into two events: one that fires every time a single agent finishes its part, and one that streams the individual words of an agent's response as they are generated. These callbacks help with logging, progress bars, real-time UIs, and any case where you need to know what is happening inside a complex workflow without waiting for the whole thing to finish. You can watch the graph execute live instead of squinting at logs after the fact.

---

## Improvements

### Execution Prompt Written Once Per Subtask

The autonomous loop used to repeat the same instruction inside its own memory on every step, which made the model think it was being asked to redo work it had already finished. The instruction is now stored exactly once per subtask, so the agent moves forward instead of looping over the same ground.

### Skip the `think` Tool When Native Thinking Is Enabled

Some newer models have built-in thinking — they reason internally before answering. When that is turned on, the framework's own "think" helper tool was just adding extra back-and-forth for no benefit, so it is now hidden whenever the model already thinks for itself.

### Modernised `arun_stream` Async Behaviour

The async streaming function used to rely on a deprecated piece of Python's async machinery and quietly hide errors that happened in the background. It now uses the modern equivalent and surfaces any error to your code, so you can actually see and handle problems instead of having them swallowed silently.

### Faster `any_to_str` in AgentRearrange

Inside AgentRearrange there is a helper that turns arbitrary values into strings. The helper now first checks if the value is already a string and skips the conversion entirely if it is. On hot paths that touch this helper many times, that small check meaningfully reduces wasted work.

### Removed `uvloop` and `winloop` Dependencies

Two optional async libraries — uvloop and winloop — were listed as dependencies but never actually used in production code paths. They have been removed along with the dead functions that referenced them, shrinking the install footprint and removing a platform-specific package nobody needed.

### Timestamps in Conversation History

When you ask a Conversation for its history as a string, every message now carries an ISO timestamp showing exactly when it happened. That makes it dramatically easier to line conversations up against application logs or debug a sequence of events after the fact.

### Senator Assembly Module Removed

A deprecated simulation module called senator_assembly — more than 3 400 lines of code, plus its example file and the line that re-exported it — has been deleted. It is the single largest reason the release ends up shorter than it started.

---

## Bug Fixes

### Non-OpenAI Providers No Longer Return Empty Strings

A regression had been causing providers other than OpenAI — including Anthropic and Google — to return blank responses whenever the reasoning_effort setting was used. The fix restores normal output across every supported provider, so you no longer have to avoid the setting on non-OpenAI models.

### Streaming Thinking Panel No Longer Corrupts the Terminal

When the framework streamed an agent's "thinking" output into a Rich live display, it sometimes ended up scrambling the terminal — overlapping panels, cursors in the wrong place. The thinking output is now flushed and rendered before the live display starts, so the terminal stays clean throughout the run.

### `_generate_final_summary` Forwards `streaming_callback`

At the very end of an autonomous run there is a final-summary step. That step was forgetting to pass through your streaming callback, so the summary tokens were generated silently even when you had asked to receive them as they arrived. The callback now flows through correctly and you see the summary stream in real time.

### License Metadata Corrected to Apache 2.0

The project's package metadata was incorrectly marking it as MIT-licensed even though the actual LICENSE file is Apache 2.0. That mismatch is now fixed in both the package configuration and the standard licence classifier, so tools and registries see the correct, consistent licence.

---

## Tests

### MEMORY.md Persistence Coverage

A dedicated test suite now covers MEMORY.md round-trip persistence, compact-with-archive behaviour, and timestamp formatting in the history string. The new memory pipeline is exercised end-to-end rather than through mocks, so regressions in this critical area get caught immediately.

### Real-LLM Streaming and Autonomous-Loop Suite

A 71-test harness across 19 classes uses real Agent and LiteLLM instances to validate the streaming pipeline, async streaming behaviour, thinking-panel rendering, autonomous-loop fixes, and final-summary callback threading. Confidence in the streaming and loop paths is now backed by actual model calls, not stubs.

---

## Docs

### Agent Memory Guide

A new guide walks through the full memory flow — persistent memory, the context compressor, and compaction with archive snapshots. It is the canonical reference for everything memory-related introduced in v12.

### Persistent Memory and Context Compression Examples

The agent reference documentation gains a section dedicated to persistent_memory and context_compression with annotated example patterns. Readers can see the combined usage at a glance without piecing it together from the changelog.

### GraphWorkflow Streaming Callback Example

A complete runnable example demonstrates the new on_node_complete and streaming_callback hooks on GraphWorkflow. Anyone wiring up DAG observability has a copy-paste starting point.

### README Updates

Several README passes correct import paths so SwarmRouter, AutoSwarmBuilder, and AOP all resolve from the top-level swarms package, and refresh model-name references and integration examples to match current usage.

---

## Conclusion

v12 is fundamentally a memory and reliability release. The combination of persistent memory, the context compressor, conversation compaction with archive snapshots, and on-disk message history transforms the agent from a stateless request handler into a long-lived, restart-safe entity capable of running indefinitely without losing context, while the new grep tool meaningfully extends what the autonomous loop can do on real codebases.

The supporting work is just as important: 7 improvements tighten the autonomous loop and shed dead dependencies, 4 bug fixes resolve real production issues across non-OpenAI providers, streaming, and the final-summary phase, and 2 new test suites — including a 71-test real-LLM harness — back the new paths with end-to-end coverage rather than mocks. Across 35 commits and a net reduction of around 1 700 lines, the framework comes out of this sprint smaller, sharper, and more capable than it went in.
