# `AdvisorSwarm`

The `AdvisorSwarm` implements the [advisor strategy](https://claude.com/blog/the-advisor-strategy) described in Anthropic's research (April 2026). It pairs a cheaper **executor** model that drives the task end-to-end with a powerful **advisor** model consulted on-demand between executor turns.

The executor runs every turn. The advisor is on-demand — consulted between executor turns when budget allows. Both agents read from and write to the same shared conversation context. The advisor never calls tools or produces user-facing output.

This is provider-agnostic: any model supported by LiteLLM works for either role.

```mermaid
graph TD
    A[User Task] --> B[Shared Context]
    B --> C{Advisor Budget?}
    C -->|Yes| D[Advisor reads context, provides guidance]
    D --> B
    C -->|No| E[Executor reads context, produces output]
    B --> E
    E --> B
    E --> F{More turns?}
    F -->|Yes| C
    F -->|No| G[Return Result]
```

The swarm follows this workflow:

1. User task goes into the shared conversation
2. Before each executor turn, the advisor reads the full shared context and provides guidance (if budget allows)
3. The executor reads the full shared context (including any advisor guidance) and produces output
4. Both advisor guidance and executor output are added to the shared conversation
5. Repeat for `max_loops` executor turns


## Key Features

| Feature | Description |
|---------|-------------|
| **Executor-Driven Loop** | The executor runs every turn — it's the main driver |
| **On-Demand Advisor** | Advisor is consulted between turns, not in a fixed sequence |
| **Shared Context** | Both agents read from and write to the same conversation |
| **Budget Control** | `max_advisor_uses` caps advisor consultations per run |
| **Provider-Agnostic** | Any LiteLLM-supported model works for either role |
| **Custom Agents** | Pass pre-configured agents with tools, MCP, or any Agent settings |


## Constructor

### `AdvisorSwarm.__init__()`

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `name` | `str` | `"AdvisorSwarm"` | No | Human-readable name |
| `description` | `str` | `"An executor-advisor swarm..."` | No | Description of the swarm's purpose |
| `executor_model_name` | `str` | `"claude-sonnet-4-6"` | No | Model for the executor agent |
| `advisor_model_name` | `str` | `"claude-opus-4-6"` | No | Model for the advisor agent |
| `executor_system_prompt` | `str` | Built-in | No | System prompt for the executor |
| `advisor_system_prompt` | `str` | Built-in | No | System prompt for the advisor |
| `max_advisor_uses` | `int` | `3` | No | Max advisor consultations per `run()`. 0 = executor runs alone. |
| `max_loops` | `int` | `1` | No | Number of executor turns |
| `output_type` | `OutputType` | `"dict-all-except-first"` | No | Format for output (dict, str, list, final, json, yaml) |
| `verbose` | `bool` | `False` | No | Enable detailed logging |
| `executor_agent` | `Agent` | `None` | No | Pre-configured Agent for execution (e.g., with tools or MCP) |
| `advisor_agent` | `Agent` | `None` | No | Pre-configured Agent for advising |
| `tools` | `List[Callable]` | `None` | No | Tools available to the executor agent only |

#### Raises

| Exception | Condition |
|-----------|-----------|
| `ValueError` | If `max_advisor_uses < 0`, `max_loops < 1`, or model names are empty |

---

## Core Methods

### `run()`

Execute the advisor-executor orchestration flow.

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `task` | `str` | — | **Yes** | The task to accomplish |
| `img` | `str` | `None` | No | Optional single image input |
| `imgs` | `List[str]` | `None` | No | Optional list of image inputs |

#### Returns

| Type | Description |
|------|-------------|
| `Any` | Formatted conversation history according to `output_type` |

---

### `batched_run()`

Run the swarm on multiple tasks sequentially.

| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `tasks` | `List[str]` | — | **Yes** | List of task strings |

#### Returns

| Type | Description |
|------|-------------|
| `List[Any]` | List of results, one per task |

---

## Usage Examples

### Basic Usage

```python
from swarms import AdvisorSwarm

swarm = AdvisorSwarm(
    executor_model_name="claude-sonnet-4-6",
    advisor_model_name="claude-opus-4-6",
    max_advisor_uses=3,
    max_loops=1,
    verbose=True,
)

result = swarm.run(
    "Write a Python function that implements binary search on a sorted list. "
    "Include proper error handling, type hints, and edge cases."
)

print(result)
```

### Multi-Turn with Advisor Guidance

Run the executor for multiple turns, with the advisor providing guidance before each:

```python
from swarms import AdvisorSwarm

swarm = AdvisorSwarm(
    executor_model_name="claude-sonnet-4-6",
    advisor_model_name="claude-opus-4-6",
    max_advisor_uses=3,
    max_loops=3,
)

result = swarm.run("Design and implement a REST API rate limiter in Python")
```

### Custom Executor with Tools

Pass a pre-configured executor agent with tools while keeping the advisor tool-free:

```python
from swarms import Agent, AdvisorSwarm


def write_file(filename: str, content: str) -> str:
    """Write content to a file."""
    with open(filename, "w") as f:
        f.write(content)
    return f"Written: {filename}"


executor = Agent(
    agent_name="Executor",
    model_name="claude-sonnet-4-6",
    max_loops=1,
    tools=[write_file],
)

swarm = AdvisorSwarm(
    executor_agent=executor,
    advisor_model_name="claude-opus-4-6",
)

result = swarm.run("Create a Python module for string manipulation utilities")
```

### Executor Only (No Advisor)

Set `max_advisor_uses=0` to run the executor alone:

```python
from swarms import AdvisorSwarm

swarm = AdvisorSwarm(
    executor_model_name="claude-sonnet-4-6",
    advisor_model_name="claude-opus-4-6",
    max_advisor_uses=0,
    max_loops=1,
)

result = swarm.run("Simple task that doesn't need advisor guidance")
```

### Different Providers

The swarm is provider-agnostic. Use any models LiteLLM supports:

```python
from swarms import AdvisorSwarm

# OpenAI models
swarm = AdvisorSwarm(
    executor_model_name="gpt-4.1-mini",
    advisor_model_name="gpt-4.1",
)

# Mix providers
swarm = AdvisorSwarm(
    executor_model_name="gpt-4.1-mini",
    advisor_model_name="claude-opus-4-6",
)
```

---

## Architecture Details

### Shared Context

Both agents read from and write to the same `Conversation` object. This mirrors the Anthropic diagram where the advisor reads the same context as the executor. On each turn:

1. The advisor reads `conversation.get_str()` — sees everything so far
2. The advisor's guidance is added to the conversation
3. The executor reads `conversation.get_str()` — sees the task, any prior output, and the advisor's guidance
4. The executor's output is added to the conversation

### Advisor Budget

The `max_advisor_uses` parameter controls how many times the advisor is consulted:

| `max_advisor_uses` | `max_loops` | Behavior |
|---|---|---|
| `0` | `1` | Executor runs alone — no advisor |
| `1` | `1` | Advisor guides once, executor runs once |
| `3` | `3` | Advisor guides before each of 3 executor turns |
| `1` | `3` | Advisor guides first turn only, executor runs 3 turns |

### Multi-Turn Execution

When `max_loops > 1`, the executor runs multiple turns. Each turn, it reads the full conversation — including its own previous output and any advisor guidance — so it can build on prior work. The advisor's budget is distributed across turns: it is consulted before each executor turn until the budget is exhausted.
