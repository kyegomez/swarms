---
name: swarms
description: Build agents and multi-agent systems with the Swarms framework — the Agent class, tools, autonomous loops, memory, and the 15+ multi-agent architectures (SequentialWorkflow, ConcurrentWorkflow, GraphWorkflow, HierarchicalSwarm, SwarmRouter, and more). Use whenever writing, reviewing, or debugging code that imports `swarms`.
---

# Swarms

Swarms is a multi-agent orchestration framework. Everything is built from one primitive — `Agent` — which multi-agent structures compose. This document is verified against **swarms v14.0.0**.

## Golden rules

1. **Import from the top level**: `from swarms import Agent`, never `from swarms.structs.agent import Agent`. The one common exception is `PlannerWorkerSwarm` (see below).
2. **Every agent needs a unique `agent_name`** — memory files and swarm routing key on it.
3. **Default to `max_loops=1`.** Use a specific integer for production. Use `"auto"` only for genuinely open-ended work.
4. **Pass `tools=None`, not `tools=[]`.** An empty list breaks schema generation.
5. **Check `examples/`** — 586 runnable examples live there. One is probably close to what you need.
6. **Never set `streaming_on=True` and `streaming_callback` together.** Pick one.

## Setup

```bash
pip install -U swarms
```

Set the key for whichever provider you use — any [LiteLLM](https://docs.litellm.ai/docs/providers) model string works:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GROQ_API_KEY="..."
export WORKSPACE_DIR="agent_workspace"   # where agent state and memory land
```

---

# Part 1 — The Agent

```python
from swarms import Agent

agent = Agent(
    agent_name="Analyst",
    agent_description="Analyzes market data and produces summaries.",
    system_prompt="You are a precise financial analyst.",
    model_name="gpt-5.4",
    max_loops=1,
)

result = agent.run("Summarize the state of the semiconductor market.")
```

`Agent.__init__` accepts 90+ parameters. These are the ones that matter:

| Parameter | Type | Default | Purpose |
|---|---|---|---|
| `agent_name` | `str` | `"swarm-worker-01"` | Unique identity; keys memory + routing |
| `agent_description` | `str` | generic | How orchestrators decide to route to it |
| `system_prompt` | `str` | built-in | Persona and instructions |
| `model_name` | `str` | `"gpt-5.4"` | Any LiteLLM model string |
| `max_loops` | `int \| "auto"` | `1` | Iterations, or autonomous mode |
| `tools` | `list[Callable]` | `None` | Python functions the agent may call |
| `temperature` | `float` | `0.5` | Sampling temperature |
| `max_tokens` | `int` | model max | Output cap per call |
| `top_p` | `float` | `None` | Nucleus sampling |
| `context_length` | `int` | `None` | Token budget; triggers compression at 90% |
| `output_type` | `str` | `"str-all-except-first"` | Return shape — see below |
| `streaming_on` | `bool` | `False` | Stream tokens to stdout |
| `streaming_callback` | `Callable` | `None` | Stream tokens to your function |
| `interactive` | `bool` | `False` | REPL — prompts the user each loop |
| `verbose` | `bool` | `False` | Debug logging |
| `print_on` | `bool` | `True` | Print the final output |
| `autosave` | `bool` | `False` | Persist agent state after each run |
| `retry_attempts` | `int` | `3` | LLM call retries |
| `reasoning_effort` | `str` | `None` | `minimal`/`low`/`medium`/`high`/`xhigh`/`ultra`/`max`/`none` |
| `thinking_tokens` | `int` | `1024` | Extended thinking budget (Claude) |
| `mcp_url` / `mcp_urls` | `str` / `list[str]` | `None` | MCP servers to load tools from |
| `handoffs` | `list[Agent]` | `None` | Agents this one may delegate to |
| `persistent_memory` | `bool` | `False` | Read/write `MEMORY.md` across restarts |
| `context_compression` | `bool` | `True` | Auto-summarize near the context limit |
| `plan_enabled` | `bool` | `False` | Plan before executing |
| `mode` | `str` | `"standard"` | `"standard"`, `"fast"`, `"interactive"` |
| `fallback_models` | `list[str]` | `None` | Models to try if the primary fails |

**`output_type` options**: `"str"`, `"list"`, `"dict"`, `"json"`, `"yaml"`, `"xml"`, `"final"`, `"last"`, `"all"`, `"basemodel"`, `"str-all-except-first"`, `"dict-all-except-first"`, `"dict-final"`, `"list-final"`.

### Running

```python
agent.run(task="...")                          # standard
agent.run(task="...", img="chart.png")         # one image
agent.run(task="...", imgs=["a.png", "b.png"]) # several images
agent.run(task="...", n=3)                     # 3 independent samples
await agent.arun("...")                        # async
```

`Agent.run` signature: `run(task=None, img=None, imgs=None, correct_answer=None, streaming_callback=None, n=1)`.

### Streaming

```python
# To stdout
agent = Agent(agent_name="Writer", model_name="gpt-5.4", streaming_on=True)
agent.run("Write a haiku about distributed systems.")

# To a callback (do NOT combine with streaming_on)
def on_token(token: str) -> None:
    print(token, end="", flush=True)

agent = Agent(agent_name="Writer", model_name="gpt-5.4", streaming_callback=on_token)
agent.run("Write a haiku.")

# Async streaming
async for token in agent.arun_stream("Explain async/await."):
    print(token, end="", flush=True)
```

---

# Part 2 — Tools

Any Python function with type hints and a docstring becomes a tool. The framework generates the OpenAI function schema automatically — **the docstring is the tool description the model reads, so write it for the model.**

```python
from swarms import Agent

def get_stock_price(ticker: str) -> str:
    """Fetch the current stock price for a ticker symbol.

    Args:
        ticker: Stock ticker symbol, e.g. 'AAPL'.

    Returns:
        The current price as a formatted string.
    """
    import yfinance as yf
    return f"{ticker}: ${yf.Ticker(ticker).fast_info['last_price']:.2f}"

agent = Agent(
    agent_name="StockAnalyst",
    model_name="gpt-5.4",
    tools=[get_stock_price],
    max_loops=3,          # needs > 1 so it can act on the tool result
)
agent.run("What are Apple and Microsoft trading at?")
```

**`max_loops` must exceed 1 for tool use** — loop 1 calls the tool, loop 2 uses the result.

Related knobs: `tool_call_summary=True` (summarize tool output), `show_tool_execution_output=True` (print raw returns), `tool_retry_attempts` (retries on tool failure).

### MCP servers

```python
agent = Agent(
    agent_name="MCPAgent",
    model_name="gpt-5.4",
    mcp_url="http://localhost:8000/sse",
    # or: mcp_urls=["http://localhost:8000/sse", "http://localhost:8001/sse"]
    max_loops=3,
)
```

Inspect what a server exposes before wiring it up:

```python
from swarms.tools.mcp_manager import MCPManager

mgr = MCPManager(mcp_url="http://localhost:8000/sse")
print(mgr.list_tool_names())
schemas = mgr.get_tools()          # aget_tools() for the async form
```

### Handoffs

Give an agent a roster it can delegate to. It receives a `handoff_task` tool automatically.

```python
triage = Agent(
    agent_name="Triage",
    model_name="gpt-5.4",
    handoffs=[billing_agent, technical_agent, refunds_agent],
    max_loops=3,
)
triage.run("My invoice is wrong and the app won't load.")
```

---

# Part 3 — Autonomous mode (`max_loops="auto"`)

The agent runs plan → execute → reflect until it decides it is finished, with **16 built-in tools** available:

| Group | Tools |
|---|---|
| Planning | `create_plan`, `think`, `subtask_done`, `complete_task`, `respond_to_user` |
| Files | `create_file`, `update_file`, `read_file`, `list_directory`, `delete_file` |
| System | `run_bash`, `grep` |
| Delegation | `create_sub_agent`, `assign_task`, `check_sub_agent_status`, `cancel_sub_agent_tasks` |

```python
agent = Agent(
    agent_name="Researcher",
    model_name="gpt-5.4",
    max_loops="auto",
    tools=[search_web],           # your tools stack on top of the built-ins
    persistent_memory=True,
    context_compression=True,
    context_length=32000,
)
agent.run("Research the top 5 vector databases and write compare.md")
```

Restrict the built-in set with `selected_tools` (default `"all"`):

```python
agent = Agent(
    agent_name="ReadOnly",
    max_loops="auto",
    selected_tools=["create_plan", "think", "read_file", "grep", "complete_task"],
)
```

Inspect the full list at runtime with `agent.get_all_selected_tools()`.

⚠️ **`run_bash` and `delete_file` are real.** In autonomous mode the agent can modify and delete files and execute shell commands. Scope `selected_tools` and set `WORKSPACE_DIR` deliberately.

---

# Part 4 — Memory and conversation

### Persistent memory

`persistent_memory=True` reads `{WORKSPACE_DIR}/agents/{agent_name}/MEMORY.md` on startup and appends to it each response. It is **off by default** — set it in every process that should share the memory.

```python
agent = Agent(agent_name="ProjectAssistant", model_name="gpt-5.4", persistent_memory=True)
agent.run("My project is called Helios. Remember that.")

# Later process, same agent_name and the flag set again → it remembers.
```

### Context compression

`context_compression=True` (default) fires at 90% of `context_length`, summarizing history in place so long sessions never hit the wall. Leave it on for anything long-running.

### Conversation

```python
from swarms import Conversation

conv = Conversation(
    name="my-conversation",     # note: `name`, not `agent_name`
    system_prompt="You are helpful.",
    time_enabled=True,
    token_count=True,
)
conv.add("user", "What is 2+2?")
conv.add("assistant", "4.")

conv.return_history_as_string()
conv.search("2+2")
conv.compact(summary="User asked arithmetic. Answer: 4.")   # archives, then collapses
conv.save_as_json("conv.json")
```

---

# Part 5 — Multi-agent architectures

## Choosing one

| Situation | Use |
|---|---|
| Single task | `Agent` |
| Linear A→B→C | `SequentialWorkflow` |
| Same task, many agents at once | `ConcurrentWorkflow` |
| Custom mix of sequential + parallel | `AgentRearrange` |
| Dependency graph / fan-out-fan-in | `GraphWorkflow` |
| Many models, one synthesized answer | `MixtureOfAgents` |
| Manager delegates to specialists | `HierarchicalSwarm` |
| Open discussion | `GroupChat` |
| Discrete decision by consensus | `MajorityVoting` |
| Quality-critical evaluation | `CouncilAsAJudge` |
| Structured adversarial debate | `DebateWithJudge` |
| Deep multi-stage research | `HeavySwarm` |
| Route each task to the best agent | `MultiAgentRouter` |
| Plan then execute with workers | `PlannerWorkerSwarm` |
| Don't know yet | `SwarmRouter(swarm_type="auto")` or `AutoSwarmBuilder` |

## SequentialWorkflow

Each agent's output becomes the next agent's context.

```python
from swarms import Agent, SequentialWorkflow

pipeline = SequentialWorkflow(
    agents=[researcher, analyst, writer],
    max_loops=1,
    output_type="dict",
)
pipeline.run("Analyze how rate hikes affect tech stocks.")
```

Options: `team_awareness=True` (agents see the roster), `multi_agent_collab_prompt=True`, `drift_detection=True`.

## ConcurrentWorkflow

All agents run the same task in parallel.

```python
from swarms import Agent, ConcurrentWorkflow

workflow = ConcurrentWorkflow(
    agents=agents,
    max_workers=5,
    show_dashboard=True,
    on_error="store",          # or "raise"
)
workflow.run("List 10 use cases for multi-agent AI.")
```

## AgentRearrange — flow DSL

```python
from swarms import Agent, AgentRearrange

pipeline = AgentRearrange(
    agents=[planner, coder, reviewer, tester],
    flow="Planner -> Coder -> Reviewer, Tester",
    max_loops=1,
)
pipeline.run("Build an email validator.")
```

- `A -> B` — sequential, B receives A's output
- `A, B` — concurrent, same input
- `A -> B, C -> D` — A, then B and C in parallel, then D on their combined output

**Every name in `flow` must match an `agent_name` in `agents`**, or it fails at run time. There is no human-in-the-loop step — split into separate `.run()` calls and insert your own `input()` between them.

## GraphWorkflow — DAG

Pass agents directly to `add_node`/`add_edge`; there is no need to wrap them in `Node` objects.

```python
from swarms import Agent, GraphWorkflow

wf = GraphWorkflow(name="research-dag", max_loops=1)

for a in (ingestion, branch_a, branch_b, merger):
    wf.add_node(a)

wf.add_edge(ingestion, branch_a)      # fan out
wf.add_edge(ingestion, branch_b)
wf.add_edge(branch_a, merger)         # fan in
wf.add_edge(branch_b, merger)

wf.set_entry_points(["Ingestion"])
wf.set_end_points(["Merger"])

def on_done(node: str, result) -> None:
    print(f"[{node}] {len(str(result))} chars")

results = wf.run(task="Analyze this dataset two ways and merge.", on_node_complete=on_done)
```

`add_node` also accepts a nested `GraphWorkflow`. Other options: `backend="networkx"|"rustworkx"`, `max_parallel_nodes`, `checkpoint_dir`, `streaming_callback`.

## SwarmRouter — one entry point

Swap architectures without rewriting orchestration.

```python
from swarms import Agent, SwarmRouter

router = SwarmRouter(agents=agents, swarm_type="SequentialWorkflow", max_loops=1)
router.run("Write a post about transformers.")
```

Valid `swarm_type` values — **exactly these 16**:

`"AgentRearrange"`, `"MixtureOfAgents"`, `"SequentialWorkflow"`, `"ConcurrentWorkflow"`, `"GroupChat"`, `"MultiAgentRouter"`, `"HierarchicalSwarm"`, `"MajorityVoting"`, `"CouncilAsAJudge"`, `"HeavySwarm"`, `"BatchedGridWorkflow"`, `"LLMCouncil"`, `"DebateWithJudge"`, `"RoundRobin"`, `"PlannerWorkerSwarm"`, `"auto"`.

`"AutoSwarmBuilder"` and `"SpreadSheetSwarm"` are **not** router types — use those classes directly. With `swarm_type="AgentRearrange"` you must also pass `rearrange_flow`.

## MixtureOfAgents

Workers answer independently; an aggregator synthesizes. Best with diverse providers.

```python
from swarms import Agent, MixtureOfAgents

moa = MixtureOfAgents(
    agents=[worker_gpt, worker_claude, worker_llama],
    aggregator_agent=aggregator,       # optional; falls back to aggregator_model_name
    layers=3,
    max_loops=1,
)
moa.run("Best practices for securing a Kubernetes cluster?")
```

## HierarchicalSwarm

A director decomposes the task, delegates, and synthesizes results.

```python
from swarms import Agent, HierarchicalSwarm

swarm = HierarchicalSwarm(
    agents=[data_worker, writing_worker, review_worker],
    director=director,            # optional; else built from director_model_name
    max_loops=2,
    planning_enabled=True,
    parallel_execution=True,
    director_feedback_on=True,
)
swarm.run("Produce a competitive analysis of the AI chip market.")
```

Also: `agent_as_judge=True`, `max_agent_retries`, `max_reassignment_attempts`, `interactive=True`.

## GroupChat

Asynchronous and self-selecting — no rounds, no speaker-selection function. Every agent scores how much it wants to speak (0–1); replies above `threshold` are broadcast. Ends at `max_loops` messages or after `idle_timeout` seconds of silence.

```python
from swarms import Agent, GroupChat

chat = GroupChat(
    agents=[optimist, pessimist, realist],   # at least 2 required
    max_loops=10,
    threshold=0.5,           # raise for a more selective room
    recency_penalty=0.3,     # discourages one agent dominating
    idle_timeout=8.0,
)
chat.run("Should we adopt AI for medical diagnosis?")
```

`auto_equip=True` (default) injects the required `RESPOND_TOOL` into every agent — **you do not need to pass it yourself**. Set `auto_equip=False` only if you attach `RESPOND_TOOL` manually via `tools_list_dictionary`.

## MajorityVoting

Agents answer independently; a consensus agent picks the winner.

```python
from swarms import Agent, MajorityVoting

mv = MajorityVoting(
    agents=voters,
    consensus_agent_model_name="gpt-5.4",
    max_loops=1,
)
mv.run("Python or Rust for a high-performance web server?")
```

## CouncilAsAJudge

Evaluates a response across dimensions. **It builds its own council from model names — it does not take an `agents` list or a `judge` agent.**

```python
from swarms import CouncilAsAJudge

council = CouncilAsAJudge(
    model_name="gpt-5.4",
    aggregation_model_name="gpt-5.4",
    random_model_name=True,
    max_loops=1,
)
council.run("Should we store biometric data on-device only?")
```

## DebateWithJudge

```python
from swarms import Agent, DebateWithJudge

debate = DebateWithJudge(
    pro_agent=pro,
    con_agent=con,
    judge_agent=judge,
    max_loops=3,          # rounds
)
debate.run("Motion: open-source LLMs will surpass closed-source by 2027.")
```

`preset_agents=True` generates pro/con/judge for you from `model_name`. The kwargs are `pro_agent`/`con_agent`/`judge_agent` — **not** `agents=[...]` plus `judge=`.

## HeavySwarm

Deep multi-stage analysis. **Configured by model names, not by an `agents` list.**

```python
from swarms import HeavySwarm

swarm = HeavySwarm(
    question_agent_model_name="gpt-5.4",
    worker_model_name="gpt-5.4",
    max_loops=1,
    timeout=900,
    show_dashboard=True,
    worker_tools=[search_web],
)
swarm.run("Analyze the implications of AGI on global labour markets.")
```

## PlannerWorkerSwarm

A planner decomposes the task and workers execute; a judge checks completion each cycle. **Not exported at the top level:**

```python
from swarms.structs.planner_worker_swarm import PlannerWorkerSwarm

swarm = PlannerWorkerSwarm(
    agents=workers,                    # workers only — the planner is built internally
    planner_model_name="gpt-5.4",
    judge_model_name="gpt-5.4",
    max_planner_depth=1,
    max_loops=1,
)
swarm.run("Build a go-to-market strategy for a B2B SaaS product.")
```

## Others

```python
from swarms import (
    MultiAgentRouter,      # routes each task to the best-fit agent
    RoundRobinSwarm,       # fixed rotation
    LLMCouncil,            # members answer, rank peers anonymously, chairman synthesizes
    BatchedGridWorkflow,   # agent i runs task i
    AutoSwarmBuilder,      # generates the agents and architecture from a description
    SpreadSheetSwarm,      # structured tabular processing
    AdvisorSwarm, SelfMoASeq, HybridHierarchicalClusterSwarm,
)

builder = AutoSwarmBuilder(name="MarketResearch", description="...", max_loops=1)
builder.run("Research the EV market and find growth opportunities.")
```

---

# Part 6 — Execution helpers

```python
from swarms import (
    run_agents_concurrently,
    run_agents_with_different_tasks,
    run_agents_concurrently_async,
    batch_agent_execution,
    run_single_agent,
    aggregate,
)

run_agents_concurrently(agents=agents, task="Summarize today's news.", max_workers=8)
run_agents_with_different_tasks([(agent_a, "task A"), (agent_b, "task B")])  # list of tuples
batch_agent_execution(agents=agents, tasks=tasks, max_workers=10)
aggregate(workers=agents, task="...", aggregator_model_name="gpt-5.4")
```

Note `run_agents_with_different_tasks` takes a **list of `(agent, task)` tuples**, not a dict.

## Scheduling

```python
from swarms import CronJob

job = CronJob(agent=agent, interval="10minutes", job_id="market-check")
job.run(task="Check for unusual market activity.")
```

`interval` is `"<number><unit>"`, and the unit must be one of `second`, `seconds`, `minute`, `minutes`, `hour`, `hours`. Abbreviations like `"30s"` raise `CronJobConfigError`, as does a zero interval.

## Loading agents from files

```python
from swarms import AgentLoader

loader = AgentLoader(concurrent=True)
agents = loader.load_agents_from_markdown("agents/")   # also: _from_yaml, _from_csv
agent = loader.load_agent_from_markdown("agents/researcher.md")
```

---

# Part 7 — Pitfalls

| Don't | Do | Why |
|---|---|---|
| `from swarms.structs.agent import Agent` | `from swarms import Agent` | Submodule paths move between versions |
| `tools=[]` | `tools=None` | Empty list breaks schema generation |
| `tools=[f]` with `max_loops=1` | `max_loops=3` | Loop 1 calls the tool; it needs loop 2 to use the result |
| Same `agent_name` on several agents | Unique names | `MEMORY.md` is keyed on it — they corrupt each other |
| `streaming_on=True` + `streaming_callback` | Pick one | They conflict |
| `CouncilAsAJudge(agents=..., judge=...)` | Model-name kwargs | It takes no `agents` or `judge` argument |
| `DebateWithJudge(agents=[p, c], judge=j)` | `pro_agent=`, `con_agent=`, `judge_agent=` | Those kwarg names don't exist |
| `HeavySwarm(num_agents=4, model_name=...)` | `question_agent_model_name=`, `worker_model_name=` | Those kwarg names don't exist |
| `from swarms import PlannerWorkerSwarm` | `from swarms.structs.planner_worker_swarm import ...` | Not exported at the top level |
| `swarm_type="AutoSwarmBuilder"` | Use the class directly | Not one of the 16 router types |
| `GraphWorkflow.add_node(Node(...))` | `add_node(agent)` | It takes the agent itself |
| Building agents inside a loop | Build once, reuse | Construction is expensive |
| `context_compression=False` on long runs | Leave it `True` | The run will hit the context wall |
| Bare `max_loops="auto"` in production | Integer `max_loops` | Autonomous runs have no natural stopping point |

## Production configuration

```python
agent = Agent(
    agent_name="ProductionAgent",
    agent_description="...",
    model_name="gpt-5.4",
    max_loops=3,
    context_length=32000,
    context_compression=True,
    persistent_memory=True,
    autosave=True,
    retry_attempts=3,
    fallback_models=["claude-sonnet-4-6"],
    verbose=False,
)
```

## Debugging

- `verbose=True` — full internal logging
- `show_tool_execution_output=True` — raw tool returns
- `output_type="all"` — the complete conversation instead of just the final message
- `agent.get_all_selected_tools()` — the autonomous tool roster
- `agent.short_memory.return_history_as_string()` — dump the conversation

---

## Reference

- Docs: [docs.swarms.world](https://docs.swarms.world) · [Agent API](https://docs.swarms.world/api/agent)
- Examples: [`examples/`](examples/) — `single_agent/`, `multi_agent/`, `tools/`, `guides/`
- Source: `swarms/structs/` (agents + swarms), `swarms/agents/` (loops, judges, routers), `swarms/tools/`
- Contributing: [CONTRIBUTING.md](CONTRIBUTING.md)
