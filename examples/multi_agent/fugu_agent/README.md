# FuguAgent

A multi-agent orchestration system that behaves like a single model API.

## Overview

FuguAgent implements the **Fugu/Trinity pattern** — a multi-agent system that presents itself as a single model API. A dedicated coordinator model dynamically assigns tasks to a ranked pool of worker agents using **tool-calling** (not text parsing).

## Architecture

```
User Task --> Coordinator --> decide_next_step() tool --> AgentTask {role, worker, instruction, visibility}
                                                                           |
                                                                           v
                                                              +------------------+
                                                              | Worker Execution |
                                                              +------------------+
                                                                           |
                                                                           v
                                                               WorkflowState
                                                               + MemoryStore
                                                                           |
                                                                           v
                                                                   Aggregation
                                                                           |
                                                                           v
                                                               Final Answer
```

## Key Features

- **Tool-calling orchestration**: The coordinator commits to structured AgentTasks via tool calls, not fragile text parsing
- **Dynamic roles**: Roles are not hardcoded — coordinator assigns whichever fits: planner, coder, researcher, writer, verifier, reviewer, etc.
- **Model capability ranking**: Workers ranked by MODEL_TIER scores; hardest tasks assigned to most capable models
- **SQLite persistent memory**: Task artifacts and metadata persist across turns and sessions
- **Visibility routing**: Each AgentTask specifies which prior step outputs the worker can see

## Installation

FuguAgent is part of the swarms package and requires API keys for the models you want to use.

Set environment variables:
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
```

## Quick Start

### Auto-detect models (simplest)

```python
from examples.multi_agent.fugu_agent import FuguAgent

agent = FuguAgent(
    coordinator_model="gpt-4o-mini",
    max_turns=5,
    verbose=True,
)

result = agent.run("Write a short story about AI discovering music.")
```

### With explicit workers

```python
from swarms import Agent
from examples.multi_agent.fugu_agent import FuguAgent

workers = [
    Agent(agent_name="coder", model_name="gpt-4o", max_loops=1),
    Agent(agent_name="researcher", model_name="claude-sonnet-4-5", max_loops=1),
    Agent(agent_name="writer", model_name="gpt-4o-mini", max_loops=1),
]

agent = FuguAgent(
    coordinator_model="gpt-4o-mini",
    workers=workers,
    max_turns=5,
)

result = agent.run("Research, write, and review an article about quantum computing.")
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `coordinator_model` | str | `"gpt-4o-mini"` | Model for the coordinator agent |
| `workers` | list[Agent] | `None` | Explicit worker agents |
| `worker_models` | list[str] | `None` | Model names to auto-create workers |
| `max_turns` | int | `5` | Maximum workflow turns before terminating |
| `confidence_threshold` | float | `0.85` | Min confidence for verification acceptance |
| `verbose` | bool | `False` | Enable verbose output |
| `memory_db_path` | str | `None` | Custom path for SQLite memory database |

## How It Works

1. **Coordinator decides**: The coordinator receives the original task, history, and memory context, then calls `decide_next_step` tool to assign a role/worker/instruction
2. **Worker executes**: The assigned worker runs with visibility into relevant prior outputs
3. **Verification**: Special roles like "verifier" or "reviewer" trigger built-in verification against accumulated work
4. **Aggregation**: After max_turns or acceptance, coordinator synthesizes all step outputs into a final answer

## Model Tiers

Workers are automatically ranked by capability:

| Tier | Models |
|------|--------|
| 10 | GPT-5, o3, o4 |
| 9 | Claude Opus 4/3, Gemini 3 Ultra |
| 8 | Claude Sonnet 4-5, Gemini 3 Pro, Llama 4 405B |
| 7 | Claude Sonnet 3-5, Gemini 2.5 Pro, GPT-4o |
| 6 | Gemini 2.5 Flash, GPT-4 Turbo, Llama 4 70B, DeepSeek R1/V3 |
| 5 | Gemini 2.0 Flash, GPT-4o-mini, Qwen 3 32B, DeepSeek R1 |
| 4 | GPT-4, Qwen 3 8B, Llama 4 8b, Gemma 3 12B |
| 3 | GPT-3.5 Turbo (and below) |

## Files

- `fugu_agent.py` — Core implementation with pydantic models and type hints
- `example_basic.py` — Minimal usage with auto-detected models
- `example_with_workers.py` — With explicit worker configuration

## Testing

Run examples directly:
```bash
python example_basic.py
python example_with_workers.py
```

## License

Part of the swarms package.
