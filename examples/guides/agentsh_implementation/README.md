# AgentSH on Swarms

A single-file implementation of **Agensh: Scaling Organizational Intelligence to 1,024 Agents** (Zhan, Song, Dong, Huang, Lian, Xia, Wei — Microsoft Research, [arXiv:2609.26781](https://arxiv.org/abs/2609.26781)), built on the Swarms `Agent`. The whole implementation is in `agentsh.py`.

AgentSH has **no orchestrator**. It starts N identical workers, and each one runs the same loop on its own thread: find something useful to do, claim it, build it, check it, and merge it. The workers coordinate only through shared infrastructure: a versioned workspace, a message channel, and a board of findings. No agent plans, assigns tasks, or combines results for the others.

## Quick start

```python
from agentsh import AgentSH

org = AgentSH(
    num_workers=8,
    model_name="gpt-5.4",
    max_turns=6,
    output_dir="output",      # write the final `main` branch to disk
)
report = org.run("Build a pure-Python Markdown-to-HTML converter with tests.")
print(report)
```

`run(task)` returns a Markdown report containing every file on `main`, the commit log, and each worker's patch summaries. After the run, you can also inspect `org.workspace`, `org.messages`, `org.context` and `org.workers` directly.

Run the built-in example with `python agentsh.py`. It needs an API key for your model provider, for example `OPENAI_API_KEY`.

## How it works

```
                 ┌──────────────── repeat ────────────────┐
                 ▼                                        │
worker 1..N:  gather context → claim → act → verify → merge
                 │               │       │       │        │
                 ▼               ▼       ▼       ▼        ▼
          ┌─────────────────┬────────────────────┬───────────────┐
          │ SharedWorkspace │ MessageInterface   │ SharedContext │
          │ main + branches │ #task channel, DMs │ typed board   │
          └─────────────────┴────────────────────┴───────────────┘
```

### 1. The cooperation loop is in the prompt, not the runtime

As in the paper (§2.3, Appendix A), every worker gets the same system prompt, differing only in its handle (`agentsh-worker1`, `agentsh-worker2`, …). `WORKER_PROMPT` describes the infrastructure, the goal, a few firm rules, and the six-step work cycle:

1. **Gather context:** read the board, the channel, and `main`.
2. **Claim:** post a `CLAIM` on the board.
3. **Act:** build on your private branch.
4. **Verify:** check the work against the goal, and revise until it passes.
5. **Merge:** call `merge_main`. If the merge is refused, `sync_main`, resolve the conflicts, re-verify, and merge again.
6. **Publish:** post a `PATCH_SUMMARY` (`files= | idea= | evidence=`), then go back to step 1.

The Python code never enforces this order. Workers decide for themselves what to claim, who to message, and when they're finished.

### 2. The infrastructure

All three components are in-memory, thread-safe classes:

| Paper component | Class | What it does |
|---|---|---|
| Shared workspace (Gitea) | `SharedWorkspace` | A Git-like file store: one `main` branch plus a private `Branch` for each worker. |
| Message interface (Mattermost) | `MessageInterface` | The `#task` channel for announcements, plus direct messages between workers. |
| Shared context (DeLM) | `SharedContext` | An append-only board of short typed entries. |

**Merging in `SharedWorkspace`.** Merges are checked file by file:

- A merge is refused if it changes a file that a peer changed on `main` since your last sync, and the two versions differ. You get `MergeResult(merged=False, conflicts=...)`.
- `sync` brings the latest `main` into your branch. Files you didn't touch update to the latest version, and your own edits are kept. Any file both you and a peer changed gets Git-style `<<<<<<< yours / ======= / >>>>>>> main` markers.
- A branch that still has conflict markers can't be merged.
- A successful merge creates a `Commit`, updates your branch to the new `main`, and posts an announcement to the channel as `@workspace`.

**The board in `SharedContext`.** Entries are one of `OBSERVED`, `FACT`, `FAIL`, `CLAIM` or `PATCH_SUMMARY`.

- Summaries are capped at 100 characters (300 for `PATCH_SUMMARY`). Longer text goes in `detail`, which peers can read with `board_unfold`.
- `board_grep` searches the full history. In a query, `,` means OR and `&` means AND: `a&b,c&d` matches (a AND b) OR (c AND d).

The data types (`BoardEntry`, `Message`, `Commit`, `Branch`, `MergeResult`) are pydantic models. All of them except `Branch` are frozen.

### 3. Workers, turns, and message delivery

A `Worker` wraps one Swarms `Agent`, which runs that worker's own tool-calling loop. Around it, `Worker` adds the worker's identity, its read positions in the channel, board and inbox, its tools, and the prompt for each turn. This follows the paper's split: the single-agent harness decides how one worker reasons, and AgentSH decides how the workers coordinate.

- **One turn is one `agent.run(prompt)` call.** Within a turn, the agent can call tools up to `loops_per_turn` times. The turn ends when the model replies with plain text and no tool call, and that text becomes the worker's status. The worker's conversation carries over from one turn to the next.
- **Two delivery paths, as in Appendix B:**
  - *Low priority, delivered at the start of each turn.* The turn prompt includes new channel posts and merge announcements (`== EVENTS ==`), direct messages that arrived between turns, and the most recent `context_window` board entries (`==== SHARED CONTEXT ====`).
  - *High priority, delivered mid-turn.* Every infrastructure tool is wrapped so that new direct messages and new board entries from peers are appended to its result under `---- arrived while you worked ----`. That's how one worker can interrupt another in the middle of a turn to avoid a collision.
- **Nudges:**
  - A worker that called no tools in its last turn gets the paper's "there is always new work" prompt.
  - A worker that wrote nothing to the board gets a reminder to publish what it found.
- **Wrap-up reminders, from Appendix C:**
  - The "stop dispatching new features" reminder is sent two turns before the end, or at 87.5% of `time_budget`.
  - The "merge anything ready, then stop" reminder is sent on the last turn, or at 98.5% of the budget.
- **Stopping:** a worker stops when it calls `declare_goal_achieved`, runs out of turns (`max_turns`), or runs past `time_budget`. The budget is checked between turns, so a turn that has already started runs to completion. If one worker's turn raises an error, it's logged and that worker continues; the other workers aren't affected.

### 4. Worker tools

Each worker has 14 infrastructure tools, plus any tools you pass with `tools=`:

| Area | Tools |
|---|---|
| Board | `board_write`, `board_read`, `board_grep`, `board_unfold` |
| Messages | `channel_post`, `direct_message`, `channel_history` |
| Workspace | `list_files`, `read_file`, `write_file`, `delete_file`, `sync_main`, `merge_main` |
| Lifecycle | `declare_goal_achieved` |

Tools you pass with `tools=` are ordinary environment tools, such as a test runner. They aren't wrapped, so peer updates don't arrive through them.

## Configuration

| Parameter | Default | Meaning |
|---|---|---|
| `name` | `"agentsh"` | Organization name; workers are named `{name}-worker{i}`. |
| `num_workers` | `4` | Number of workers running at once. |
| `model_name` | `"gpt-5.4"` | Any LiteLLM model string, shared by every worker. |
| `max_turns` | `6` | Turns per worker. |
| `loops_per_turn` | `12` | Maximum model/tool steps within one turn. |
| `time_budget` | `None` | Wall-clock budget in seconds. |
| `stagger_seconds` | `0.0` | Delay between starting each worker, to reduce early collisions (the paper used 30s, then 3s). |
| `context_window` | `40` | Number of recent board entries in each turn prompt. |
| `channel_window` | `30` | Maximum number of unseen channel messages in each turn prompt. |
| `tools` | `None` | Extra tools for every worker. |
| `output_dir` | `None` | Directory to export `main` to after the run. |
| `print_on` | `False` | Print each agent's output. |
| `agent_kwargs` | `None` | Extra keyword arguments for every `Agent`, for example `{"temperature": 0.3}`. |

Workers are built with `tool_call_summary=False`. Otherwise Swarms makes an extra model call after every tool call. You can turn it back on with `agent_kwargs={"tool_call_summary": True}`.

Each call to `run()` starts a fresh organization: new infrastructure and new workers with empty conversations.

## Differences from the paper

- **In-memory infrastructure instead of real services.** The paper uses Gitea and Mattermost. Here the workspace, messages and board live in memory, and `main` is written to disk only if you set `output_dir`. Merge conflicts are detected per file, not per line of text.
- **Threads, not a cluster.** Each worker is a thread in one process. The paper ran 1,024 agents across 16 machines, so very large organizations will need a distributed runtime.
- **Turns instead of a persistent event loop.** Direct messages and board entries arrive mid-turn only through tool results. As in the paper, they can't interrupt a tool that's already running.
- **General-purpose tasks.** The prompt is written for any task, not just ProgramBench. Workers can only read and write files unless you give them tools to run code, tests, or anything else they need to check their work.
- **No idle timer.** There's no 10-minute idle detector. Its nudge is sent at the start of the next turn instead.
