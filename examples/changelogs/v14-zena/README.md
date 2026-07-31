# Swarms v14 "Zena" — Blog Examples

Runnable versions of every code example from [Swarms v14 "Zena"](../../content/blog/swarms-v14-zena-release.mdx).

Each file is self-contained and maps to one section of the post.

## Setup

```bash
pip install -U swarms
export OPENAI_API_KEY=sk-...
```

A few examples reference other providers (`claude-sonnet-4-6` in the hierarchical swarm, Ollama in the MCP example). Set those keys too, or edit the `model_name` values.

## Examples

| File | Blog section | What it shows |
|---|---|---|
| [01_opentelemetry_tracing.py](01_opentelemetry_tracing.py) | Observability | Spans across a workflow, and the one switch that turns telemetry off |
| [02_instrument_custom_harness.py](02_instrument_custom_harness.py) | Observability | `capture_init` + `@trace_run` on your own harness, with context-propagating threads |
| [03_mcp_manager.py](03_mcp_manager.py) | Unified MCP Manager | One server, many servers, headers, timeouts, and OAuth token storage |
| [04_auto_agent_builder.py](04_auto_agent_builder.py) | AutoAgentBuilder | Roster generation, `max_agents` vs `num_agents`, feeding a workflow |
| [05_auction_swarm.py](05_auction_swarm.py) | AuctionSwarm | Agents bidding on a task; default and custom scoring |
| [06_groupchat_turn_based.py](06_groupchat_turn_based.py) | Turn-Based GroupChat | Single-speaker bidding with a recency penalty |
| [07_hierarchical_swarm_recovery.py](07_hierarchical_swarm_recovery.py) | HierarchicalSwarm | Worker retry, task reassignment, planning, judge, director overrides |
| [08_computer_use_tools.py](08_computer_use_tools.py) | Computer-Use Tools | Full toolset, and a read-only subset |
| [09_class_to_pydantic.py](09_class_to_pydantic.py) | Pydantic Schemas | Schema from a constructor, and the round-trip back to an `Agent` |
| [10_concurrent_workflow_on_error.py](10_concurrent_workflow_on_error.py) | ConcurrentWorkflow | `on_error` failure policy |
| [11_graph_workflow.py](11_graph_workflow.py) | Performance | Fan-out/fan-in DAG on native rustworkx |
| [12_thread_pool_sizing.py](12_thread_pool_sizing.py) | Performance | `max_workers` across concurrent harnesses |
| [13_agent_loader_csv.py](13_agent_loader_csv.py) | CSV Agents | Loading agents from CSV via `AgentLoader` |

## Two things worth knowing before you run these

**`max_agents` is a ceiling, not a target.** `AutoAgentBuilder` is instructed to prefer the smallest roster that covers a task, so `max_agents=5` on a three-role problem returns three agents. That is correct behavior. Use `num_agents` when the count is a requirement — `04_auto_agent_builder.py` runs both side by side.

**`13_agent_loader_csv.py` needs an `agents.csv`.** It is the only example that reads an external file. Every other file runs as-is.

## Cost

These make real API calls. `04`, `05`, and `12` each run several agents, and `06` runs a multi-turn group chat capped at 12 messages. Lower `max_loops`, `max_agents`, or swap to a cheaper model if you are just kicking the tires.
