# FuguAgent: Multi-Agent System as a Single Model

## Description

New `FuguAgent` class in `examples/multi_agent/fugu_agent/` implementing the Fugu/Trinity orchestration pattern — a multi-agent system that presents itself as a single model API.

> **Note:** This implementation lives in `examples/` and is not exported from the main `swarms` package. It is available for testing and evaluation before potential inclusion.

### Core Architecture

The `FuguAgent` coordinates a pool of worker agents through a dedicated coordinator model using **tool-calling** (not text parsing). At each step the coordinator calls the `decide_next_step` tool, committing to a structured `AgentTask {role, worker, instruction, visibility}`. The result is stored directly via closure capture, bypassing fragile history parsing.

**Key components:**

- **`decide_next_step` tool** — Function-call based orchestration. The coordinator decides role, worker, instruction, and visibility for each step and commits via the tool. No JSON text parsing required.
- **Dynamic roles** — Roles are not hardcoded. The coordinator assigns whichever role fits: `planner`, `researcher`, `coder`, `writer`, `verifier`, `reviewer`, `executor`, `summarizer`, etc.
- **Model capability ranking** — Workers are ranked by `MODEL_TIER` scores. The coordinator's system prompt lists workers by tier, ensuring the most powerful models handle the hardest subtasks.
- **`MemoryStore`** — SQLite-backed persistent memory across turns and sessions.
- **Visibility routing** — Each `AgentTask` specifies which prior step outputs (by index) the worker can see, implementing the Conductor's access-list pattern.
- **Chain-of-thought aggregation** — Final answer synthesized by passing all step outputs through the coordinator.

**Files:**

| File | Description |
|------|-------------|
| `examples/multi_agent/fugu_agent/fugu_agent.py` | Core FuguAgent implementation with pydantic models and type hints |
| `examples/multi_agent/fugu_agent/__init__.py` | Module exports |
| `examples/multi_agent/fugu_agent/example_basic.py` | Minimal usage with auto-detected models |
| `examples/multi_agent/fugu_agent/example_with_workers.py` | Usage with explicit worker agents |
| `examples/single_agent/fugu_example.py` | Deprecated redirect to new location |

## Architecture

```mermaid
flowchart TD
    User(["User Task"]) --> Coordinator

    Coordinator -->|"decide_next_step()"| Tool
    Tool -->|"AgentTask JSON"| Holder
    Holder --> T1

    subgraph T1[" "]
        direction LR
        T2["_decide_holder read"]
    end

    T2 --> ExecStep["execute_step()"]
    ExecStep --> ExRole{role}
    ExRole -->|planner / researcher / coder / writer| Execute
    ExRole -->|verifier / reviewer| Verify
    ExRole -->|any| Context["build visibility context"]

    subgraph Worker_Pool["Worker Pool"]
        W1["[7] general (gpt-4o)"]
        W2["[5] coder (gpt-4o-mini)"]
        W3["[5] researcher (claude-sonnet)"]
    end

    Execute --> Context
    Context --> W1
    Context --> W2
    Context --> W3

    W1 --> R1[/Result/]
    W2 --> R2[/Result/]
    W3 --> R3[/Result/]

    R1 --> WS[WorkflowState]
    R2 --> WS
    R3 --> WS

    Verify --> VResult{ver.accept?}
    VResult -->|ACCEPT| Done
    VResult -->|REVISE| Coordinator

    WS --> Aggregator
    Done --> Aggregator["coordinator.aggregate()"]
    Aggregator --> Final(["Final Answer"])

    subgraph Mem["MemoryStore (SQLite)"]
        M1["per-turn artifacts"]
        M2["session context"]
    end

    WS -.-> M1
    M1 -.-> M2
```

### Execution Loop

```mermaid
sequenceDiagram
    participant User
    participant Coord as Coordinator
    participant Tool
    participant Fugu
    participant Worker
    participant Verifier

    User->>Fugu: run(task)

    loop max_turns
        Fugu->>Coord: coordinator.run(history_ctx + memory)
        Coord->>Tool: decide_next_step(role, worker, instruction, visibility)
        Tool->>Tool: store task in _decide_holder
        Tool-->>Coord: AgentTask JSON
        Coord-->>Fugu: run() returns

        Fugu->>Fugu: agent_task = _decide_holder.pop()

        alt agent_task.role in verifier / reviewer
            Fugu->>Verifier: run(accumulated_work)
            Verifier-->>Fugu: ACCEPT or REVISE + diagnosis
            Fugu->>Fugu: if accept: break
        else
            Fugu->>Worker: run(instruction + visibility_context)
            Worker-->>Fugu: result
        end

        Fugu->>Fugu: WorkflowState.results.append()
        Fugu->>Fugu: MemoryStore.save(turn)
    end

    Fugu->>Coord: coordinator.run(synthesis_prompt)
    Coord-->>Fugu: Final Answer
    Fugu-->>User: Final Answer
```

## Usage

```python
from examples.multi_agent.fugu_agent import FuguAgent

agent = FuguAgent(
    coordinator_model="gpt-4o-mini",
    max_turns=5,
    verbose=True,
)

result = agent.run("Write a short story about a robot discovering music.")
```

Workers are auto-detected from `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY`, or can be passed explicitly:

```python
from swarms import Agent
from examples.multi_agent.fugu_agent import FuguAgent

agent = FuguAgent(
    workers=[
        Agent(agent_name="coder", model_name="gpt-4o"),
        Agent(agent_name="researcher", model_name="claude-sonnet-4-5"),
    ],
    max_turns=5,
)
```

## Code Quality

- **Pydantic models** for `AgentTask`, `AgentTaskResult`, `VerificationResult` with field validation
- **Comprehensive type hints** throughout all methods and functions
- **Docstrings** on all public classes and methods
- **`__repr__`** methods on `WorkflowState`, `MemoryStore`, and `FuguAgent` for debugging

## Dependencies

- `pydantic` — used for data validation on `AgentTask`, `AgentTaskResult`, `VerificationResult`
- Standard library: `sqlite3`, `json`, `os`, `time`

## Tag Maintainer

kye@swarms.world
