# GroupChat Comprehensive Examples

!!! abstract "Overview"
    This guide showcases the asynchronous, self-selecting `GroupChat`. Each
    agent listens to broadcast messages in parallel and independently decides
    whether to reply. Replies whose score exceeds `threshold` are broadcast
    back into the room. The chat ends when either `max_loops` messages have
    been posted or no new message arrives for `idle_timeout` seconds.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Basic Setup](#basic-setup)
3. [Threshold Tuning](#threshold-tuning)
4. [Stopping Conditions](#stopping-conditions)
5. [Output Types](#output-types)
6. [Conversation History](#conversation-history)
7. [Batched Execution](#batched-execution)
8. [Complete Example: Marketing Team](#complete-example-marketing-team)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## Prerequisites

!!! info "Before You Begin"
    Make sure you have:

    - Python 3.10+ installed
    - A valid API key for your model provider(s)
    - The Swarms package installed

```bash
pip install swarms
```

## Basic Setup

### Environment Configuration

```python
# .env file
OPENAI_API_KEY="your-api-key-here"
```

### Import Required Modules

`GroupChat` and the forced `RESPOND_TOOL` schema are both re-exported from the
top-level `swarms` package.

```python
from dotenv import load_dotenv
import os
from swarms import Agent, GroupChat, RESPOND_TOOL
```

### Creating Example Agents

Every agent participating in a `GroupChat` **must** be configured with
`tools_list_dictionary=[RESPOND_TOOL]`. The recommended defaults are
`max_loops=1` (one LLM call per inbox message) and `persistent_memory=False`
(decisions are made from shared chat history, not per-agent memory files).

```python
analyst = Agent(
    agent_name="analyst",
    system_prompt="You are a data analyst. You excel at analyzing data, creating charts, and providing insights.",
    model_name="gpt-4.1",
    max_loops=1,
    persistent_memory=False,
    tools_list_dictionary=[RESPOND_TOOL],
)

researcher = Agent(
    agent_name="researcher",
    system_prompt="You are a research specialist. You are great at gathering information, fact-checking, and providing detailed research.",
    model_name="gpt-4.1",
    max_loops=1,
    persistent_memory=False,
    tools_list_dictionary=[RESPOND_TOOL],
)

writer = Agent(
    agent_name="writer",
    system_prompt="You are a content writer. You excel at writing clear, engaging content and summarizing information.",
    model_name="gpt-4.1",
    max_loops=1,
    persistent_memory=False,
    tools_list_dictionary=[RESPOND_TOOL],
)

reviewer = Agent(
    agent_name="reviewer",
    system_prompt="You are a quality reviewer. You ensure accuracy, completeness, and quality of all outputs.",
    model_name="gpt-4.1",
    max_loops=1,
    persistent_memory=False,
    tools_list_dictionary=[RESPOND_TOOL],
)

agents = [analyst, researcher, writer, reviewer]
```

### Minimal Run

```python
chat = GroupChat(
    name="Research Team",
    description="A team for collaborative analysis.",
    agents=agents,
    max_loops=10,
    threshold=0.5,
    idle_timeout=8.0,
)

task = "Let's create a comprehensive market analysis report."
response = chat.run(task)
print(response)
```

A typical run produces only a subset of agents speaking — those whose
self-scored desire to reply exceeded `threshold`. Agents that have nothing
substantive to add stay silent rather than padding the conversation.

## Threshold Tuning

The `threshold` parameter is the single most important dial for shaping
behaviour. It sets the minimum self-score an agent's reply must achieve before
the reply is broadcast.

### Selective panel (high threshold)

!!! example "High Threshold — Quiet, Expert Room"
    Raising the threshold makes agents reply only when they are confident the
    contribution is genuinely new.

```python
selective_chat = GroupChat(
    name="Selective Panel",
    description="Only high-confidence replies are broadcast.",
    agents=agents,
    max_loops=10,
    threshold=0.75,
)

task = "Evaluate the trade-offs of vector databases versus knowledge graphs."
response = selective_chat.run(task)
print(response)
```

### Brainstorming chat (low threshold)

!!! example "Low Threshold — Lively Brainstorm"
    Lowering the threshold encourages more participation, useful for ideation.

```python
brainstorm_chat = GroupChat(
    name="Brainstorm",
    description="Loosely scored — most agents will weigh in.",
    agents=agents,
    max_loops=20,
    threshold=0.3,
)

task = "Brainstorm marketing angles for a developer tooling launch."
response = brainstorm_chat.run(task)
print(response)
```

## Stopping Conditions

`GroupChat` stops when **either** of these is true:

- `max_loops` total messages have been posted (the initial user task counts as
  message one).
- No new message has been posted for `idle_timeout` seconds.

### Hard cap on messages

```python
capped_chat = GroupChat(
    name="Capped Chat",
    description="Hard message cap.",
    agents=agents,
    max_loops=6,        # stop after 6 messages total
    threshold=0.5,
)
capped_chat.run("Outline the steps for a product launch.")
```

### Faster idle shutdown

```python
quick_idle_chat = GroupChat(
    name="Quick Idle",
    description="Shut down faster when the room is quiet.",
    agents=agents,
    max_loops=20,
    threshold=0.5,
    idle_timeout=3.0,   # 3 seconds of silence ends the chat
)
quick_idle_chat.run("Suggest names for a new internal AI project.")
```

## Output Types

`output_type` is forwarded to `history_output_formatter`. The most useful
values are `"str-all-except-first"` (default), `"dict"`, `"list"`, and `"json"`.

### Dictionary Output

```python
dict_chat = GroupChat(
    name="Dict Output Team",
    description="A team with dictionary output",
    agents=agents,
    output_type="dict",
)

response = dict_chat.run("Let's discuss the project plan.")
print(type(response))   # <class 'list'> or <class 'dict'> depending on formatter
print(response)
```

### String Output

```python
string_chat = GroupChat(
    name="String Output Team",
    description="A team with string output",
    agents=agents,
    output_type="str-all-except-first",
)

response = string_chat.run("Let's discuss the project plan.")
print(type(response))   # <class 'str'>
print(response)
```

## Conversation History

The full chat is recorded on `chat.conversation`. It can be inspected after
the run (or between runs) using the standard `Conversation` API.

```python
chat = GroupChat(
    name="History Team",
    description="A team with conversation history",
    agents=agents,
    max_loops=10,
    threshold=0.5,
)

chat.run("Let's discuss the first topic.")
chat.run("Now let's discuss the second topic.")

# Full history as a printable string
print(chat.conversation.return_history_as_string())

# Raw list of message dicts
for msg in chat.conversation.messages:
    print(msg)
```

## Batched Execution

`run_batch` runs the same `GroupChat` instance sequentially against a list of
tasks and returns one formatted history per task.

```python
chat = GroupChat(
    name="Batch Team",
    description="A team for batch processing",
    agents=agents,
    max_loops=10,
    threshold=0.5,
)

tasks = [
    "Analyze the Q1 sales data.",
    "Research market trends for Q2.",
    "Draft a summary report.",
]

responses = chat.run_batch(tasks)
for i, response in enumerate(responses, 1):
    print(f"Task {i} Response:\n{response}\n" + "=" * 50)
```

## Complete Example: Marketing Team

!!! success "Full Implementation"
    A complete example showcasing a multi-provider self-selecting team.

```python
from dotenv import load_dotenv
import os
from swarms import Agent, GroupChat, RESPOND_TOOL

load_dotenv()

market_researcher = Agent(
    agent_name="market_researcher",
    system_prompt="You are a market research specialist. You analyze market trends, customer behavior, and competitive landscape.",
    model_name="gpt-4.1",
    max_loops=1,
    persistent_memory=False,
    tools_list_dictionary=[RESPOND_TOOL],
)

content_strategist = Agent(
    agent_name="content_strategist",
    system_prompt="You are a content strategist. You create engaging content strategies and messaging frameworks.",
    model_name="claude-sonnet-4-6",
    max_loops=1,
    persistent_memory=False,
    tools_list_dictionary=[RESPOND_TOOL],
)

data_analyst = Agent(
    agent_name="data_analyst",
    system_prompt="You are a data analyst. You analyze campaign performance, metrics, and ROI data.",
    model_name="gpt-4.1",
    max_loops=1,
    persistent_memory=False,
    tools_list_dictionary=[RESPOND_TOOL],
)

creative_director = Agent(
    agent_name="creative_director",
    system_prompt="You are a creative director. You oversee creative vision and ensure brand consistency.",
    model_name="gpt-4.1",
    max_loops=1,
    persistent_memory=False,
    tools_list_dictionary=[RESPOND_TOOL],
)

agents = [
    market_researcher,
    content_strategist,
    data_analyst,
    creative_director,
]

marketing_chat = GroupChat(
    name="Marketing Strategy Team",
    description="A collaborative team for developing marketing strategies",
    agents=agents,
    max_loops=12,
    threshold=0.55,
    idle_timeout=10.0,
    output_type="str-all-except-first",
)

task = (
    "We're planning the launch of a new B2B developer tool. "
    "Identify the target market, outline a messaging strategy, "
    "back it up with relevant metrics, and ensure the creative "
    "direction stays consistent across channels."
)

print("Starting marketing strategy session...")
response = marketing_chat.run(task)

print("\n" + "=" * 80)
print("MARKETING STRATEGY SESSION RESULTS")
print("=" * 80)
print(response)

print("\n" + "=" * 80)
print("CONVERSATION HISTORY")
print("=" * 80)
print(marketing_chat.conversation.return_history_as_string())
```

A typical run will have only the agents whose contribution is in-scope
actually post — for example the market researcher and data analyst may
dominate early, then the content strategist and creative director chime in.
Agents that have nothing new to add remain silent.

## Best Practices

### 1. Agent Design

!!! tip "Agent Best Practices"
    - **Clear roles**: Give each agent a distinct, well-defined role so the
      decision prompt can tell which messages fall inside its expertise.
    - **Descriptive names**: Use clear, descriptive agent names; they appear
      in every other agent's decision prompt.
    - **Focused prompts**: Keep system prompts focused on the agent's
      expertise — wide-scope prompts make agents reply to everything.
    - **Always pass `RESPOND_TOOL`**: `tools_list_dictionary=[RESPOND_TOOL]`
      is required; without it the agent cannot return a structured score.

### 2. Threshold tuning

!!! tip "Threshold Guidelines"
    - **0.7+** — selective expert panel; few replies, all high-value.
    - **0.5 (default)** — balanced discussion.
    - **0.3** — lively brainstorming where most agents weigh in.

### 3. Stopping conditions

!!! tip "Bounding the conversation"
    - Always set a sensible `max_loops` (e.g., 10–20 for chats, 2–4 for
      quick consensus checks).
    - Tune `idle_timeout` to the latency of the slowest model in the room —
      it should be long enough that a slow reply isn't mistaken for silence,
      and short enough that a finished conversation actually stops.

### 4. Cross-provider rooms

!!! tip "Mixing model providers"
    Each agent runs in its own coroutine and calls its own model. There's
    nothing stopping you from mixing GPT, Claude, Llama, and Gemini in the
    same room — `LiteLLM` normalizes the calls.

### 5. Performance Optimization

!!! tip "Performance Tips"
    - Use cheap models for "supporting" agents and reserve premium models
      for the agents whose contributions you actually rely on.
    - Lower `max_loops` for latency-sensitive use cases.
    - For independent tasks, prefer `run_batch` over manually looping over
      `run` so the chat state is reset between tasks.

## Troubleshooting

!!! warning "Common Problems and Solutions"


| Issue Area                  | Common Problem                                     | Solution/Check                                                                 |
|-----------------------------|----------------------------------------------------|--------------------------------------------------------------------------------|
| API & Setup                 | API keys not working                               | Check API keys are set in the environment for every provider you use.          |
| Agent Configuration         | `ValueError: GroupChat requires at least 2 agents.` | Pass at least two agents to the constructor.                                   |
| Missing tool schema         | All agents stay silent / score always 0.0          | Confirm every agent has `tools_list_dictionary=[RESPOND_TOOL]`.                |
| Conversation never stops    | Runs to `max_loops` every time                     | Raise `threshold` so fewer replies clear the bar, or lower `max_loops`.        |
| Conversation stops too early | Idle timer fires while a slow model is still thinking | Increase `idle_timeout`.                                                       |
| Repetitive replies          | Agents echo each other                             | Tighten system prompts to discourage agreement-only replies and raise `threshold`. |
| Off-topic responses         | Agents stray off-topic                             | Improve agent system prompts — narrow the stated expertise.                    |

## Additional Resources

| Resource                     | Description                                      | Link                                                                                   |
|------------------------------|--------------------------------------------------|----------------------------------------------------------------------------------------|
| GroupChat API Reference      | Reference for the GroupChat system               | [View](https://docs.swarms.world/swarms/structs/group_chat/)                 |
| GroupChat Guide              | Step-by-step guide for GroupChat                 | [View](https://docs.swarms.world/swarms/examples/groupchat_comprehensive_examples/)    |
| Agent Documentation          | Reference for building and using agents          | [View](https://docs.swarms.world/swarms/structs/agent/)                      |
| Multi-Agent Architectures    | Concepts and architectures for multi-agent swarms| [View](https://docs.swarms.world/swarms/concept/swarm_architectures/)        |


## Connect With Us

Join our community for support, updates, and insights:

| Platform | Description | Link |
| --- | --- | --- |
| Documentation | Official documentation | [docs.swarms.world](https://docs.swarms.world) |
| Discord | Community support | [Join Discord](https://discord.gg/EamjgSaEQf) |
| Twitter | Latest news | [@swarms_corp](https://x.com/swarms_corp) |
| GitHub | Source code | [kyegomez/swarms](https://github.com/kyegomez/swarms) |
