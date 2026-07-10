# CLAUDE.md — Swarms Framework Guide

This file teaches you how to build agents and multi-agent systems with the **Swarms** framework. Read it before writing any code in this repo.

---

## Installation & Setup

```bash
pip install swarms
```

Set your LLM API key as an environment variable before running:

```bash
export OPENAI_API_KEY="sk-..."        # OpenAI / GPT models
export ANTHROPIC_API_KEY="sk-ant-..." # Claude models
export GROQ_API_KEY="..."             # Groq
# Any provider supported by LiteLLM works
```

All imports come from the top-level `swarms` package:

```python
from swarms import (
    Agent,
    SequentialWorkflow,
    ConcurrentWorkflow,
    AgentRearrange,
    GraphWorkflow,
    SwarmRouter,
    MixtureOfAgents,
    HierarchicalSwarm,
    GroupChat,
    MajorityVoting,
    # ...
)
```

---

## Project Layout

```
swarms/
├── swarms/
│   ├── structs/         # All agent + multi-agent structures (61 files)
│   │   ├── agent.py             # Core Agent class
│   │   ├── conversation.py      # Conversation / memory management
│   │   ├── sequential_workflow.py
│   │   ├── concurrent_workflow.py
│   │   ├── agent_rearrange.py
│   │   ├── graph_workflow.py
│   │   ├── swarm_router.py      # Single-entry-point router
│   │   ├── mixture_of_agents.py
│   │   ├── hiearchical_swarm.py
│   │   ├── groupchat.py
│   │   ├── majority_voting.py
│   │   ├── council_as_judge.py
│   │   ├── debate_with_judge.py
│   │   ├── heavy_swarm.py
│   │   ├── round_robin.py
│   │   ├── planner_worker_swarm.py
│   │   ├── auto_swarm_builder.py
│   │   └── multi_agent_exec.py  # run_agents_concurrently + friends
│   ├── tools/           # Tool utilities, MCP, schema conversion
│   └── utils/           # Logging, formatting helpers
├── examples/            # 586 runnable examples
│   ├── single_agent/
│   ├── multi_agent/
│   ├── tools/
│   └── guides/
└── v12_examples/        # New v12 feature examples
```

Look in `examples/` first before writing new code — there is almost certainly an existing example close to what you need.

---

## Core Primitive: Agent

`Agent` is the single building block everything else composes. All multi-agent structures wrap one or more `Agent` instances.

### Minimal agent

```python
from swarms import Agent

agent = Agent(
    agent_name="Analyst",
    model_name="gpt-5.4",
    max_loops=1,
)

result = agent.run("Summarise the current state of LLM research.")
print(result)
```

### Key constructor parameters

| Parameter | Type | Default | Purpose |
|---|---|---|---|
| `agent_name` | str | `"swarm-worker-01"` | Unique name — used for memory file paths |
| `agent_description` | str | generic | Shown to orchestrators for routing |
| `system_prompt` | str | built-in | The agent's persona / instructions |
| `model_name` | str | `"gpt-5.4"` | Any LiteLLM model string |
| `max_loops` | int \| `"auto"` | `1` | Loops before returning; `"auto"` = autonomous until done |
| `tools` | list[Callable] | `None` | Python functions the agent can call |
| `streaming_on` | bool | `False` | Stream tokens to stdout |
| `interactive` | bool | `False` | REPL mode — prompt user for input each loop |
| `context_length` | int | `None` | Token budget; triggers compression at 90 % |
| `context_compression` | bool | `True` | Auto-summarise when near context limit (v12) |
| `persistent_memory` | bool | `True` | Read/write MEMORY.md across restarts (v12) |
| `temperature` | float | `0.5` | Sampling temperature |
| `max_tokens` | int | `4096` | Max tokens per LLM call |
| `reasoning_effort` | str | `None` | `"low"`, `"medium"`, `"high"` for reasoning models |
| `thinking_tokens` | int | `None` | Extended thinking budget (Claude) |
| `output_type` | str | `"str-all-except-first"` | How to format returned output |
| `mcp_url` | str | `None` | MCP server URL to load tools from |
| `handoffs` | list | `None` | Agents this agent can hand off to |
| `plan_enabled` | bool | `False` | Generate a plan before execution |
| `autosave` | bool | `False` | Save agent state to disk after each run |

### Autonomous loop (`max_loops="auto"`)

When `max_loops="auto"` the agent runs a plan→execute→reflect loop until it decides it is done. It automatically gets access to:
- A `think` tool (disabled when `thinking_tokens` is set)
- A `grep` tool for searching files (v12)
- Bash / file tools if configured

```python
agent = Agent(
    agent_name="Researcher",
    model_name="gpt-5.4",
    max_loops="auto",
    interactive=False,
)
result = agent.run("Research the top 5 vector databases and compare them.")
```

### Model names

Use any LiteLLM-compatible string:

```python
# OpenAI
model_name="gpt-5.4"
model_name="gpt-5.4-mini"
model_name="o3"

# Anthropic
model_name="claude-opus-4-7-20251001"
model_name="claude-sonnet-4-6"
model_name="claude-haiku-4-5-20251001"

# Groq
model_name="groq/llama-3.3-70b-versatile"

# Google
model_name="gemini/gemini-2.5-pro"
```

### Running with images

```python
result = agent.run(
    task="Describe what you see in this chart.",
    img="path/to/chart.png",   # or base64 string or URL
)
```

---

## Memory & Persistence (v12)

### `persistent_memory=True` (default)

On startup the agent reads `{workspace}/agents/{agent_name}/MEMORY.md` and injects it as a system preamble. On each response it appends to that file. State survives process restarts automatically.

```python
agent = Agent(
    agent_name="ProjectAssistant",
    model_name="gpt-5.4",
    persistent_memory=True,   # default
)
# First run: agent has no prior context
agent.run("My project is called Helios. Remember that.")

# New process, same agent_name → agent remembers "Helios"
agent2 = Agent(agent_name="ProjectAssistant", model_name="gpt-5.4")
agent2.run("What is my project called?")
```

### `persistent_memory=False`

Fully stateless — no disk reads or writes. Use for short, isolated tasks where carry-over would be harmful.

```python
agent = Agent(
    agent_name="OneShot",
    model_name="gpt-5.4",
    persistent_memory=False,
)
```

### `context_compression=True` (default)

`ContextCompressor` fires automatically when token usage crosses 90 % of `context_length`. It summarises and rewrites `MEMORY.md` in place so long sessions never hit the context wall.

```python
agent = Agent(
    agent_name="LongSession",
    model_name="gpt-5.4",
    context_length=32000,
    context_compression=True,   # default
)
```

### Conversation.compact()

Manually collapse history to a single summary; creates a timestamped archive before rewriting:

```python
from swarms.structs.conversation import Conversation

conv = Conversation(agent_name="MyAgent", system_prompt="You are helpful.")
conv.add("user", "Tell me about X")
conv.add("assistant", "X is ...")

# Collapse history, archive the full log
conv.compact(summary="User asked about X. Assistant explained X.")
```

---

## Tools

### Python functions as tools

Decorate any Python function with a docstring — the framework converts it to an OpenAI function-calling schema automatically:

```python
import yfinance as yf
from swarms import Agent

def get_stock_price(ticker: str) -> str:
    """Fetch the current stock price for a given ticker symbol.

    Args:
        ticker: Stock ticker symbol, e.g. 'AAPL'.

    Returns:
        Current price as a formatted string.
    """
    data = yf.Ticker(ticker)
    price = data.fast_info["last_price"]
    return f"{ticker}: ${price:.2f}"

agent = Agent(
    agent_name="StockAnalyst",
    model_name="gpt-5.4",
    tools=[get_stock_price],
    max_loops=3,
)
result = agent.run("What is the current price of Apple and Microsoft?")
```

### Multiple tools

```python
agent = Agent(
    agent_name="ResearchAgent",
    model_name="gpt-5.4",
    tools=[search_web, get_stock_price, read_file, write_file],
    max_loops="auto",
)
```

### Tool schema from Pydantic

```python
from swarms.tools.pydantic_to_json import base_model_to_openai_function
from pydantic import BaseModel

class WeatherQuery(BaseModel):
    city: str
    units: str = "celsius"

schema = base_model_to_openai_function(WeatherQuery)
```

---

## Streaming

### Stream to stdout

```python
agent = Agent(
    agent_name="Writer",
    model_name="gpt-5.4",
    streaming_on=True,
)
agent.run("Write a short poem about distributed systems.")
```

### Stream tokens to a callback

```python
def handle_token(token: str) -> None:
    print(token, end="", flush=True)

agent = Agent(
    agent_name="Writer",
    model_name="gpt-5.4",
    streaming_callback=handle_token,
)
agent.run("Write a haiku.")
```

### Async streaming (`arun_stream`)

```python
import asyncio
from swarms import Agent

agent = Agent(agent_name="AsyncWriter", model_name="gpt-5.4", streaming_on=True)

async def main():
    async for token in agent.arun_stream("Explain async/await in Python."):
        print(token, end="", flush=True)

asyncio.run(main())
```

---

## Multi-Agent Structures

### Sequential Workflow

Agents execute **one after another**. The output of each agent is passed as context to the next.

```python
from swarms import Agent, SequentialWorkflow

researcher = Agent(agent_name="Researcher", model_name="gpt-5.4", max_loops=1)
analyst   = Agent(agent_name="Analyst",    model_name="gpt-5.4", max_loops=1)
writer    = Agent(agent_name="Writer",     model_name="gpt-5.4", max_loops=1)

pipeline = SequentialWorkflow(
    agents=[researcher, analyst, writer],
    max_loops=1,
)
result = pipeline.run("Analyse the impact of interest rate hikes on tech stocks.")
```

**When to use:** Linear pipelines where each step depends on the prior step's output. Research → Analysis → Report. Extraction → Transformation → Load.

---

### Concurrent Workflow

All agents run **in parallel** on the same task. Results are collected and returned together.

```python
from swarms import Agent, ConcurrentWorkflow

agents = [
    Agent(agent_name=f"Worker-{i}", model_name="gpt-5.4", max_loops=1)
    for i in range(5)
]

workflow = ConcurrentWorkflow(agents=agents)
results = workflow.run("List 10 use cases for multi-agent AI systems.")
```

**When to use:** Independent subtasks that can run simultaneously. Analysing multiple documents. Querying multiple data sources. Generating multiple creative variants.

---

### AgentRearrange — Flow DSL

Define execution flow as a string using a simple DSL. Mix sequential (`->`) and parallel (`,`) execution.

```python
from swarms import Agent, AgentRearrange

planner  = Agent(agent_name="Planner",  model_name="gpt-5.4", max_loops=1)
coder    = Agent(agent_name="Coder",    model_name="gpt-5.4", max_loops=1)
reviewer = Agent(agent_name="Reviewer", model_name="gpt-5.4", max_loops=1)
tester   = Agent(agent_name="Tester",   model_name="gpt-5.4", max_loops=1)

pipeline = AgentRearrange(
    agents=[planner, coder, reviewer, tester],
    flow="Planner -> Coder -> Reviewer, Tester",
    #        sequential  ↑      parallel  ↑
    max_loops=1,
)
result = pipeline.run("Build a Python function that validates email addresses.")
```

**Flow DSL rules:**
- `A -> B` — A runs, then B receives A's output
- `A, B` — A and B run concurrently with the same input
- `A -> B, C -> D` — A runs first, then B and C run concurrently, then D receives their combined output

`AgentRearrange` has no built-in human-in-the-loop step — every name in `flow` must correspond to an agent in `agents`, or the flow will fail at run time. For a human checkpoint, break the pipeline into separate `AgentRearrange`/`Agent.run()` calls and insert your own logic (e.g. `input()`) between them — see the "Human-in-the-loop with AgentRearrange" pattern below.

**When to use:** Any workflow where you need explicit, readable control over agent execution order and parallelism.

---

### GraphWorkflow — DAG Execution

Full directed-acyclic-graph (DAG) execution. Nodes are agents; edges are dependencies. Topological sort ensures correct order. Supports per-node callbacks and token streaming.

```python
from swarms import Agent, GraphWorkflow, Node, Edge, NodeType

# Build agents
analyst  = Agent(agent_name="Analyst",  model_name="gpt-5.4-mini", max_loops=1)
writer   = Agent(agent_name="Writer",   model_name="gpt-5.4-mini", max_loops=1)
reviewer = Agent(agent_name="Reviewer", model_name="gpt-5.4-mini", max_loops=1)
publisher = Agent(agent_name="Publisher", model_name="gpt-5.4-mini", max_loops=1)

# Build graph
wf = GraphWorkflow()
wf.add_node(Node(id="analyst",   type=NodeType.AGENT, agent=analyst))
wf.add_node(Node(id="writer",    type=NodeType.AGENT, agent=writer))
wf.add_node(Node(id="reviewer",  type=NodeType.AGENT, agent=reviewer))
wf.add_node(Node(id="publisher", type=NodeType.AGENT, agent=publisher))

wf.add_edge(Edge(source="analyst",  target="writer"))
wf.add_edge(Edge(source="writer",   target="reviewer"))
wf.add_edge(Edge(source="reviewer", target="publisher"))

wf.set_entry_points(["analyst"])
wf.set_end_points(["publisher"])

# Run with callbacks
def on_done(node_name: str, result: str) -> None:
    print(f"[{node_name}] finished — {len(result)} chars")

results = wf.run(
    task="Produce a market report on AI chips.",
    on_node_complete=on_done,          # fires after each node
    streaming_callback=lambda tok: print(tok, end="", flush=True),
)
```

**Diamond / fan-out fan-in pattern:**

```python
# analyst feeds both writer AND researcher concurrently,
# then editor combines both outputs
wf.add_edge(Edge(source="analyst",    target="writer"))
wf.add_edge(Edge(source="analyst",    target="researcher"))
wf.add_edge(Edge(source="writer",     target="editor"))
wf.add_edge(Edge(source="researcher", target="editor"))
```

**When to use:** Complex dependency graphs, fan-out/fan-in patterns, when you need precise control over which agents depend on which.

---

### SwarmRouter — Single Entry Point

`SwarmRouter` is the highest-level abstraction. Pass it a list of agents and a `swarm_type` — it handles the rest. Use this when you want to switch architectures without rewriting orchestration code.

```python
from swarms import Agent, SwarmRouter

agents = [
    Agent(agent_name="Analyst",  model_name="gpt-5.4", max_loops=1),
    Agent(agent_name="Writer",   model_name="gpt-5.4", max_loops=1),
    Agent(agent_name="Reviewer", model_name="gpt-5.4", max_loops=1),
]

router = SwarmRouter(
    agents=agents,
    swarm_type="SequentialWorkflow",   # swap to any SwarmType below
    max_loops=1,
)
result = router.run("Write a blog post about transformer architectures.")
```

**All `swarm_type` options:**

| SwarmType | Behaviour |
|---|---|
| `"SequentialWorkflow"` | Agents run one after another |
| `"ConcurrentWorkflow"` | Agents run in parallel |
| `"AgentRearrange"` | Flow-DSL based execution |
| `"MixtureOfAgents"` | Workers + aggregator layer |
| `"HierarchicalSwarm"` | Boss delegates to workers |
| `"GroupChat"` | Multi-agent round-table discussion |
| `"MultiAgentRouter"` | Task routed to best-fit agent |
| `"MajorityVoting"` | Agents vote; majority wins |
| `"CouncilAsAJudge"` | Council deliberates; judge decides |
| `"DebateWithJudge"` | Agents debate; judge rules |
| `"HeavySwarm"` | Intensive multi-loop deep analysis |
| `"RoundRobin"` | Round-robin task distribution |
| `"PlannerWorkerSwarm"` | Planner + worker delegation |
| `"BatchedGridWorkflow"` | Grid-based batch execution |
| `"LLMCouncil"` | LLM-based council decisions |
| `"AutoSwarmBuilder"` | Auto-configures everything |
| `"auto"` | Router selects swarm_type automatically |

---

### MixtureOfAgents

Multiple **worker** agents each respond to the task independently, then an **aggregator** agent synthesises all responses into a final answer. Repeat for multiple layers.

```python
from swarms import Agent, MixtureOfAgents

workers = [
    Agent(agent_name="Worker-GPT",    model_name="gpt-5.4",       max_loops=1),
    Agent(agent_name="Worker-Claude", model_name="claude-sonnet-4-6", max_loops=1),
    Agent(agent_name="Worker-Llama",  model_name="groq/llama-3.3-70b-versatile", max_loops=1),
]

aggregator = Agent(
    agent_name="Aggregator",
    model_name="gpt-5.4",
    system_prompt="Synthesise the following expert responses into one coherent answer.",
    max_loops=1,
)

moa = MixtureOfAgents(
    agents=workers,
    aggregator_agent=aggregator,
    layers=2,        # run worker→aggregate cycle this many times
    max_loops=1,
)
result = moa.run("What are the best practices for securing a Kubernetes cluster?")
```

**When to use:** High-stakes tasks where you want multiple independent perspectives merged into a consensus. Works especially well with diverse model providers.

---

### HierarchicalSwarm

A director agent breaks the task into subtasks and delegates them to worker agents. Workers report back; director synthesises.

```python
from swarms import Agent, HierarchicalSwarm

director = Agent(
    agent_name="Director",
    agent_description="Breaks complex tasks into subtasks and delegates them.",
    model_name="gpt-5.4",
    max_loops=1,
)

workers = [
    Agent(agent_name="DataWorker",    model_name="gpt-5.4-mini", max_loops=1),
    Agent(agent_name="WritingWorker", model_name="gpt-5.4-mini", max_loops=1),
    Agent(agent_name="ReviewWorker",  model_name="gpt-5.4-mini", max_loops=1),
]

swarm = HierarchicalSwarm(
    director=director,
    agents=workers,
    max_loops=2,
)
result = swarm.run("Produce a comprehensive competitive analysis of the AI chip market.")
```

**When to use:** Tasks naturally decomposed into subtasks where a coordinator must manage work allocation and synthesis.

---

### GroupChat

An asynchronous, self-selecting groupchat. There are no rounds or speaker-selection functions — every agent listens in parallel and decides on its own whether to chime in. A forced `respond(score, message)` function call asks each agent how much it wants to speak (0..1); replies above `threshold` are broadcast. The chat ends when `max_loops` messages have been posted or no message arrives for `idle_timeout` seconds.

```python
from swarms import Agent
from swarms.structs.groupchat import GroupChat, RESPOND_TOOL

# Every agent MUST carry RESPOND_TOOL so the chat can ask it whether to speak.
# Recommended per-agent: max_loops=1, persistent_memory=False.
optimist = Agent(
    agent_name="Optimist",
    system_prompt="You argue for the benefits.",
    model_name="gpt-5.4",
    max_loops=1,
    persistent_memory=False,
    tools_list_dictionary=[RESPOND_TOOL],
)
pessimist = Agent(
    agent_name="Pessimist",
    system_prompt="You argue for the risks.",
    model_name="gpt-5.4",
    max_loops=1,
    persistent_memory=False,
    tools_list_dictionary=[RESPOND_TOOL],
)
realist = Agent(
    agent_name="Realist",
    system_prompt="You seek balanced analysis.",
    model_name="gpt-5.4",
    max_loops=1,
    persistent_memory=False,
    tools_list_dictionary=[RESPOND_TOOL],
)

chat = GroupChat(
    agents=[optimist, pessimist, realist],
    max_loops=10,        # hard cap on total messages posted
    threshold=0.5,       # min decision score (0..1) to publish a reply
    idle_timeout=8.0,    # seconds of silence before stopping
)
result = chat.run("Should we adopt AI for medical diagnosis?")
```

**Tuning:** raise `threshold` for a more selective room; lower it for livelier chats. Raise `idle_timeout` if agents need time to think before replying.

---

### MajorityVoting

All agents independently answer the task. The answer that appears in the majority of responses wins.

```python
from swarms import Agent, MajorityVoting

voters = [
    Agent(agent_name=f"Voter-{i}", model_name="gpt-5.4-mini", max_loops=1)
    for i in range(5)
]

mv = MajorityVoting(agents=voters, max_loops=1)
result = mv.run("Is Python or Rust better for building a high-performance web server?")
```

**When to use:** Classification, yes/no decisions, or any task with a discrete answer set where you want noise reduction through consensus.

---

### CouncilAsAJudge

A council of agents each deliberate, then a judge agent makes the final ruling based on the council's reasoning.

```python
from swarms import Agent, CouncilAsAJudge

council = [
    Agent(agent_name="Expert-Security", model_name="gpt-5.4", max_loops=1),
    Agent(agent_name="Expert-Privacy",  model_name="gpt-5.4", max_loops=1),
    Agent(agent_name="Expert-Legal",    model_name="gpt-5.4", max_loops=1),
]

judge = Agent(
    agent_name="Judge",
    system_prompt="Given the council's analysis, deliver a final verdict.",
    model_name="gpt-5.4",
    max_loops=1,
)

council_swarm = CouncilAsAJudge(
    agents=council,
    judge=judge,
    max_loops=1,
)
result = council_swarm.run("Should we store user biometric data on-device only?")
```

---

### DebateWithJudge

Two or more agents argue opposing positions for multiple rounds. A judge delivers a verdict at the end.

```python
from swarms import Agent, DebateWithJudge

pro  = Agent(agent_name="Pro",  system_prompt="Argue strongly in favour.",  model_name="gpt-5.4", max_loops=1)
con  = Agent(agent_name="Con",  system_prompt="Argue strongly against.",    model_name="gpt-5.4", max_loops=1)

judge = Agent(
    agent_name="Judge",
    system_prompt="Evaluate the debate and deliver an objective verdict.",
    model_name="gpt-5.4",
    max_loops=1,
)

debate = DebateWithJudge(
    agents=[pro, con],
    judge=judge,
    max_loops=3,   # 3 rounds of argument
)
result = debate.run("Motion: Open-source LLMs will surpass closed-source models by 2027.")
```

---

### HeavySwarm

Intensive multi-loop analysis. Each agent runs for many loops on the problem, producing deep reasoning. Best for research-grade analysis.

```python
from swarms import HeavySwarm

swarm = HeavySwarm(
    num_agents=4,
    model_name="gpt-5.4",
    loops_per_agent=5,       # each agent reasons for 5 loops
    show_output=True,
)
result = swarm.run("Derive a novel approach to solving the alignment problem in AI.")
```

Or via `SwarmRouter`:

```python
from swarms import Agent, SwarmRouter

agents = [Agent(agent_name=f"Deep-{i}", model_name="gpt-5.4", max_loops=5) for i in range(4)]
router = SwarmRouter(agents=agents, swarm_type="HeavySwarm")
result = router.run("Deep analysis: implications of AGI on global labour markets.")
```

---

### RoundRobinSwarm

Distributes tasks to agents in a fixed rotation. Each agent handles every Nth task.

```python
from swarms import Agent, RoundRobinSwarm

agents = [
    Agent(agent_name=f"Handler-{i}", model_name="gpt-5.4-mini", max_loops=1)
    for i in range(3)
]

rr = RoundRobinSwarm(agents=agents, max_loops=1)

tasks = ["Task A", "Task B", "Task C", "Task D", "Task E", "Task F"]
for task in tasks:
    result = rr.run(task)
```

---

### PlannerWorkerSwarm

A planner agent generates a structured plan; worker agents execute each step.

```python
from swarms import Agent, PlannerWorkerSwarm

planner = Agent(
    agent_name="Planner",
    system_prompt="You create detailed, step-by-step execution plans.",
    model_name="gpt-5.4",
    max_loops=1,
)

workers = [
    Agent(agent_name=f"Worker-{i}", model_name="gpt-5.4-mini", max_loops=2)
    for i in range(4)
]

swarm = PlannerWorkerSwarm(
    planner_agent=planner,
    worker_agents=workers,
    max_loops=1,
)
result = swarm.run("Build a complete go-to-market strategy for a B2B SaaS product.")
```

---

### AutoSwarmBuilder

Pass a high-level description of the task — the framework automatically creates the agents, assigns roles, and runs the appropriate swarm architecture.

```python
from swarms import AutoSwarmBuilder

builder = AutoSwarmBuilder(
    name="MarketResearchSwarm",
    description="A swarm that produces comprehensive market research reports",
    max_loops=2,
)
result = builder.run("Research the electric vehicle market and identify growth opportunities.")
```

**When to use:** Rapid prototyping, when you don't know yet which structure fits, or when you want the LLM to decide.

---

## Utility Execution Helpers

```python
from swarms.structs.multi_agent_exec import (
    run_agents_concurrently,
    run_agents_concurrently_async,
    run_agents_with_different_tasks,
    run_single_agent,
)

# Same task, all agents in parallel
results = run_agents_concurrently(agents=agents, task="Summarise the news today.")

# Different task per agent
task_map = {agent: task for agent, task in zip(agents, tasks)}
results = run_agents_with_different_tasks(task_map)

# Async version
import asyncio
results = asyncio.run(run_agents_concurrently_async(agents=agents, task="..."))
```

---

## Async Support

```python
import asyncio
from swarms import Agent

agent = Agent(agent_name="AsyncAgent", model_name="gpt-5.4")

async def main():
    # Standard async run
    result = await agent.arun("What is the capital of France?")
    print(result)

    # Streaming async run
    async for token in agent.arun_stream("Explain quantum entanglement."):
        print(token, end="", flush=True)

asyncio.run(main())
```

---

## MCP Tool Integration

Load tools from any MCP server. The agent auto-discovers available tools on startup.

```python
from swarms import Agent

# Single MCP server
agent = Agent(
    agent_name="MCPAgent",
    model_name="gpt-5.4",
    mcp_url="http://localhost:8000/sse",   # SSE endpoint
    max_loops="auto",
)

# Multiple MCP servers
agent = Agent(
    agent_name="MultiMCPAgent",
    model_name="gpt-5.4",
    mcp_urls=[
        "http://localhost:8000/sse",
        "http://localhost:8001/sse",
    ],
    max_loops="auto",
)

result = agent.run("Use the available tools to complete the task.")
```

Fetch tools manually:

```python
from swarms.tools.mcp_client_tools import get_mcp_tools_sync, aget_mcp_tools

tools = get_mcp_tools_sync(server_url="http://localhost:8000/sse")

import asyncio
tools = asyncio.run(aget_mcp_tools(server_url="http://localhost:8000/sse"))
```

---

## Conversation Management

`Conversation` manages message history with optional disk persistence.

```python
from swarms.structs.conversation import Conversation

conv = Conversation(
    system_prompt="You are a helpful assistant.",
    agent_name="MyAgent",      # keys MEMORY.md to this name
    time_enabled=True,         # include ISO timestamps in history
)

conv.add("user", "What is 2+2?")
conv.add("assistant", "4.")

# Get history as string (includes timestamps in v12)
history_str = conv.return_history_as_string()

# Compact + archive
conv.compact(summary="User asked basic arithmetic. Answer: 4.")

# Pass to an agent
agent = Agent(
    agent_name="MyAgent",
    model_name="gpt-5.4",
    # agent reads MEMORY.md automatically when persistent_memory=True
)
```

---

## Choosing the Right Structure

| Situation | Use |
|---|---|
| Simple single task | `Agent` |
| Linear A→B→C pipeline | `SequentialWorkflow` |
| Same task, many agents at once | `ConcurrentWorkflow` |
| Custom mix of sequential + parallel | `AgentRearrange` |
| Complex dependency graph / DAG | `GraphWorkflow` |
| Need per-node callbacks or streaming | `GraphWorkflow` |
| Multiple models, one synthesised answer | `MixtureOfAgents` |
| Manager delegates to specialists | `HierarchicalSwarm` |
| Open discussion / brainstorming | `GroupChat` |
| Discrete decision via consensus | `MajorityVoting` |
| High-stakes ruling with deliberation | `CouncilAsAJudge` |
| Structured adversarial debate | `DebateWithJudge` |
| Deep research, many loops | `HeavySwarm` |
| Don't know yet / rapid prototyping | `AutoSwarmBuilder` or `SwarmRouter(swarm_type="auto")` |
| Need to switch architectures easily | `SwarmRouter` |

---

## Common Patterns & Recipes

### Pattern: Research → Write → Review pipeline

```python
from swarms import Agent, SequentialWorkflow

pipeline = SequentialWorkflow(agents=[
    Agent(agent_name="Researcher", system_prompt="You research topics thoroughly.", model_name="gpt-5.4"),
    Agent(agent_name="Writer",     system_prompt="You write clear, engaging content.", model_name="gpt-5.4"),
    Agent(agent_name="Editor",     system_prompt="You improve clarity and fix errors.", model_name="gpt-5.4"),
], max_loops=1)

result = pipeline.run("Write an article about the history of neural networks.")
```

### Pattern: Fan-out to specialists, fan-in to synthesiser

```python
from swarms import Agent, MixtureOfAgents

specialists = [
    Agent(agent_name="TechExpert",    system_prompt="Analyse the technical aspects.", model_name="gpt-5.4"),
    Agent(agent_name="BusinessExpert",system_prompt="Analyse the business aspects.", model_name="gpt-5.4"),
    Agent(agent_name="LegalExpert",   system_prompt="Analyse the legal aspects.",   model_name="gpt-5.4"),
]
synthesiser = Agent(agent_name="Synthesiser", model_name="gpt-5.4",
                    system_prompt="Combine expert analyses into one coherent report.")

moa = MixtureOfAgents(agents=specialists, aggregator_agent=synthesiser)
result = moa.run("Evaluate the risks of launching a fintech product in the EU.")
```

### Pattern: Autonomous agent with tools and memory

```python
import os
from swarms import Agent

def search_web(query: str) -> str:
    """Search the web for a query and return results."""
    # your implementation
    ...

def write_file(filename: str, content: str) -> str:
    """Write content to a file."""
    with open(filename, "w") as f:
        f.write(content)
    return f"Written to {filename}"

agent = Agent(
    agent_name="AutonomousResearcher",
    model_name="gpt-5.4",
    max_loops="auto",
    tools=[search_web, write_file],
    persistent_memory=True,
    context_compression=True,
    context_length=32000,
)
agent.run("Research the top 10 open-source LLMs and write a comparison report to report.md")
```

### Pattern: Multi-model ensemble with streaming

```python
import sys
from swarms import Agent, ConcurrentWorkflow

agents = [
    Agent(agent_name="GPT",    model_name="gpt-5.4",          max_loops=1),
    Agent(agent_name="Claude", model_name="claude-sonnet-4-6", max_loops=1),
    Agent(agent_name="Gemini", model_name="gemini/gemini-2.5-pro", max_loops=1),
]

workflow = ConcurrentWorkflow(agents=agents)
results = workflow.run("What is the most important unsolved problem in mathematics?")

for agent_name, answer in results.items():
    print(f"\n=== {agent_name} ===\n{answer}")
```

### Pattern: Human-in-the-loop with AgentRearrange

`AgentRearrange` has no native human-in-the-loop step — chain separate `.run()` calls yourself and insert your own checkpoint logic between them:

```python
from swarms import Agent

drafter  = Agent(agent_name="Drafter",  model_name="gpt-5.4")
finisher = Agent(agent_name="Finisher", model_name="gpt-5.4")

draft = drafter.run("Draft a press release about our product launch.")

print(f"\nAgent says:\n{draft}\n")
feedback = input("Your feedback: ")

result = finisher.run(f"Revise this draft based on the feedback.\n\nDraft:\n{draft}\n\nFeedback:\n{feedback}")
```

### Pattern: GraphWorkflow with fan-out / fan-in

```python
from swarms import Agent, GraphWorkflow, Node, Edge, NodeType

ingestion = Agent(agent_name="Ingestion", model_name="gpt-5.4-mini", max_loops=1)
branch_a  = Agent(agent_name="BranchA",   model_name="gpt-5.4-mini", max_loops=1)
branch_b  = Agent(agent_name="BranchB",   model_name="gpt-5.4-mini", max_loops=1)
merger    = Agent(agent_name="Merger",    model_name="gpt-5.4",      max_loops=1)

wf = GraphWorkflow()
for a in [ingestion, branch_a, branch_b, merger]:
    wf.add_node(Node(id=a.agent_name, type=NodeType.AGENT, agent=a))

wf.add_edge(Edge(source="Ingestion", target="BranchA"))
wf.add_edge(Edge(source="Ingestion", target="BranchB"))
wf.add_edge(Edge(source="BranchA",   target="Merger"))
wf.add_edge(Edge(source="BranchB",   target="Merger"))

wf.set_entry_points(["Ingestion"])
wf.set_end_points(["Merger"])

results = wf.run(task="Process this dataset from two angles and merge the findings.")
```

---

## What to Avoid

**Don't import from submodules directly** — always import from `swarms`:
```python
# Wrong
from swarms.structs.agent import Agent

# Right
from swarms import Agent
```

**Don't set `max_loops="auto"` without a clear stopping condition** — the agent will loop until it decides it is done or hits a resource limit. Prefer explicit `max_loops=N` for production tasks.

**Don't give all agents the same `agent_name`** — `persistent_memory` and `MEMORY.md` are keyed on `agent_name`. Duplicate names cause agents to share and corrupt each other's memory.

**Don't instantiate heavyweight structures inside tight loops** — create agents and workflows once, reuse them across calls.

**Don't pass `tools=[]` (empty list)** — pass `tools=None` instead. An empty list can confuse schema generation.

**Don't use `streaming_on=True` and `streaming_callback` together on the same agent** — `streaming_on` streams to stdout; `streaming_callback` streams to your function. Pick one.

**Don't set `context_compression=False` on very long autonomous sessions** — without compression the agent will eventually hit the context limit and raise an error.

**For long-running autonomous agents in production**, always set:
```python
agent = Agent(
    ...
    persistent_memory=True,
    context_compression=True,
    context_length=32000,
    autosave=True,
)
```
