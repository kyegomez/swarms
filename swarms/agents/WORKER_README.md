Worker pool prototype

Overview

This package contains a prototype implementation of a parallel worker execution system suitable for running many focused workers against a single git repository.

Quickstart

1. From the repository root run:

```bash
python -m swarms.agents.quickstart_worker
```

2. Telemetry is exposed at http://127.0.0.1:8008/metrics

Design notes

- `TaskQueue` (sqlite) stores tasks and supports optimistic claiming and stuck-task reclamation.
- `Worker` executes a single task using an `executor(payload, repo_path)` callback, commits changes, and pushes.
- `GitClient` wraps simple git operations and conflict detection.
- `WorkerPool` spawns thread-based workers, runs a monitor thread that reclaims stuck tasks, respawns dead worker threads, and calls `JudgeAgent` to trigger fresh-start resets when necessary.

Next steps

- Replace thread-based workers with process/container based workers for true isolation.
- Add tests that assert behavior under synthetic conflicts and long-running tasks.
- Harden security and authentication for remote git pushes.
