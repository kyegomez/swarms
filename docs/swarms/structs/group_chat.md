# GroupChat Documentation

`GroupChat` is an **asynchronous, self-selecting** multi-agent chat. Every agent
listens to every broadcast message in parallel. For each message, every other
agent independently scores how much it wants to reply. When the score clears the
configured threshold the reply is broadcast to the rest of the room. There is no
round-robin, no turn order, and no central coordinator deciding who speaks.

The chat ends when **either** of the following is true:

- `max_loops` total messages have been posted (the initial user task counts as
  the first message).
- No new message has been published for `idle_timeout` seconds.

## Features

| Feature | Description |
|---------|-------------|
| **Asynchronous fan-out** | Each agent runs its own coroutine and processes its own inbox. |
| **Self-selection via `RESPOND_TOOL`** | Every agent is forced to call `respond(score, message)` so speaking decisions are structured, not free-form text. |
| **Threshold-gated replies** | Only replies whose score exceeds `threshold` are broadcast back into the room. |
| **Idle-timeout shutdown** | The chat stops automatically after `idle_timeout` seconds of silence. |
| **Hard message cap** | `max_loops` bounds the total number of posted messages. |
| **Mixed model providers** | Workers may run on different model providers concurrently. |
| **Configurable output format** | `output_type` is passed straight through to `history_output_formatter`. |

## Installation

```bash
pip install swarms
```

## Required agent configuration

Every agent passed to `GroupChat` **must** be configured with
`tools_list_dictionary=[RESPOND_TOOL]`. The forced tool call is what produces
the `(score, message)` decision the chat depends on.

Recommended per-agent defaults:

- `max_loops=1` — one LLM call per inbox message.
- `persistent_memory=False` — every decision is made from the shared chat
  history, not from local per-agent memory files.

## Methods Reference

### Constructor (`__init__`)

**Description:**
Initializes a new `GroupChat` instance.

**Arguments:**

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `name` | str | Human-readable name used in logs and serialized state. | `"dynamic-groupchat"` |
| `description` | str | Short description of the chat. | `"Agents choose whether to speak at any time."` |
| `agents` | List[Agent] | Participating agents. At least two are required. Every agent must include `tools_list_dictionary=[RESPOND_TOOL]`. | `None` |
| `max_loops` | int | Maximum number of messages posted before stopping (including the initial task). | `20` |
| `threshold` | float | Minimum decision score (0..1) required to publish a reply. | `0.5` |
| `idle_timeout` | float | Seconds of inactivity before the chat stops. | `8.0` |
| `output_type` | str | Format passed to `history_output_formatter` (e.g., `"str-all-except-first"`, `"dict"`, `"list"`, `"json"`). | `"str-all-except-first"` |
| `verbose` | bool | Whether to emit internal log messages. | `False` |
| `print_on` | bool | Whether to print each broadcast to the terminal as a panel. | `True` |

**Raises:**

- `ValueError` — if fewer than two agents are provided.

**Example:**

```python
from swarms import Agent, GroupChat, RESPOND_TOOL

researcher = Agent(
    agent_name="Researcher",
    system_prompt="You are a research-minded agent who values evidence.",
    model_name="gpt-4.1",
    max_loops=1,
    persistent_memory=False,
    tools_list_dictionary=[RESPOND_TOOL],
)

skeptic = Agent(
    agent_name="Skeptic",
    system_prompt="You push back on weak claims and ask sharp questions.",
    model_name="gpt-4.1",
    max_loops=1,
    persistent_memory=False,
    tools_list_dictionary=[RESPOND_TOOL],
)

builder = Agent(
    agent_name="Builder",
    system_prompt="You turn ideas into concrete next steps.",
    model_name="gpt-4.1",
    max_loops=1,
    persistent_memory=False,
    tools_list_dictionary=[RESPOND_TOOL],
)

chat = GroupChat(
    name="memory-design-chat",
    description="Debate options for agent memory architecture.",
    agents=[researcher, skeptic, builder],
    max_loops=10,
    threshold=0.6,
    idle_timeout=8.0,
)

result = chat.run(
    "Should we use vector databases or knowledge graphs for agent memory?"
)
print(result)
```

## The `RESPOND_TOOL`

`RESPOND_TOOL` is a forced function-calling schema attached to every agent.
On every message the agent sees, it is required to call `respond(score, message)`:

| Field | Type | Meaning |
|-------|------|---------|
| `score` | float in `[0, 1]` | How much the agent wants to speak. `0` = silent, `1` = strongly wants to. |
| `message` | str | The reply to broadcast. Empty string if the agent chose to stay silent. |

`GroupChat` broadcasts the reply only when `score > threshold` **and** the
message is non-empty. Raising `threshold` makes agents more selective; lowering
it produces livelier chats.

Agents are encouraged to stay silent by default. They should only reply when
they can add something substantive — agreement, paraphrase, and pile-on are
explicitly discouraged in the decision prompt.

## How a turn works

1. The user task is broadcast to every agent's inbox and recorded in the
   shared conversation as the first message.
2. Each agent's coroutine pulls the message, snapshots the current chat
   history, and is asked (via the forced `respond` tool) whether to reply.
3. If the agent's score exceeds `threshold` and the message is non-empty, the
   reply is added to the shared conversation and fanned out to every other
   agent's inbox.
4. Multiple agents may decide to speak on the same message; their replies all
   land in the room and each triggers another round of decisions.
5. The conversation ends when `max_loops` is reached or no message has been
   posted for `idle_timeout` seconds.

## Advanced Examples

### Multi-provider expert panel

```python
from swarms import Agent, GroupChat, RESPOND_TOOL

data_analyst = Agent(
    agent_name="Data-Analyst",
    system_prompt="You analyze numerical data and patterns.",
    model_name="gpt-4.1",
    max_loops=1,
    persistent_memory=False,
    tools_list_dictionary=[RESPOND_TOOL],
)

market_expert = Agent(
    agent_name="Market-Expert",
    system_prompt="You provide market insights and trends.",
    model_name="claude-sonnet-4-6",
    max_loops=1,
    persistent_memory=False,
    tools_list_dictionary=[RESPOND_TOOL],
)

strategy_advisor = Agent(
    agent_name="Strategy-Advisor",
    system_prompt="You formulate strategic recommendations.",
    model_name="gpt-4.1",
    max_loops=1,
    persistent_memory=False,
    tools_list_dictionary=[RESPOND_TOOL],
)

analysis_team = GroupChat(
    name="Market Analysis Team",
    description="Comprehensive market analysis group",
    agents=[data_analyst, market_expert, strategy_advisor],
    max_loops=15,
    threshold=0.5,
)

history = analysis_team.run("""
    Analyze the current market conditions:
    1. Identify key trends
    2. Evaluate risks
    3. Recommend investment strategy
""")
```

### Batched execution

`run_batch` runs the same `GroupChat` instance against a list of tasks
sequentially and returns one formatted history per task.

```python
tasks = [
    "Analyze tech sector trends",
    "Evaluate real estate market",
    "Review commodity prices",
    "Assess global economic indicators",
]

histories = analysis_team.run_batch(tasks)

for task, history in zip(tasks, histories):
    print(f"\nAnalysis for: {task}")
    print(history)
```

## Best Practices

| Category            | Recommendations                                                                                   |
|---------------------|--------------------------------------------------------------------------------------------------|
| **Agent Design**    | Give agents clear, specific roles. Use detailed system prompts. Always pass `tools_list_dictionary=[RESPOND_TOOL]`. |
| **Per-agent setup** | Keep `max_loops=1` and `persistent_memory=False` so each decision is made fresh from shared chat history. |
| **Threshold tuning** | Raise `threshold` (e.g., `0.7`) to keep the chat focused. Lower it (e.g., `0.3`) for brainstorming-style activity. |
| **Stopping conditions** | Set `max_loops` for a hard cap on chat length, and `idle_timeout` so the chat stops on its own when the topic is exhausted. |
| **Error Handling**  | Wrap `chat.run(...)` in try/except. Failed agent decisions are logged and treated as "stay silent". |
| **Performance**     | Use cheaper models for less-specialized agents. Mix providers freely — every agent runs in its own coroutine. |

## API Reference

### GroupChat Methods

| Method | Description | Arguments | Returns |
|--------|-------------|-----------|---------|
| `run` | Run the groupchat synchronously until idle or `max_loops` is reached. | `task: str` | Formatted conversation history (type controlled by `output_type`). |
| `run_batch` | Run the chat sequentially on a list of tasks. | `tasks: List[str]` | List of formatted histories — one per task. |

### Agent Methods

| Method | Description | Returns |
|--------|-------------|---------|
| `run` | Process single task | str |
| `generate_response` | Generate LLM response | str |
| `save_context` | Save conversation context | None |
