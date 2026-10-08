# Auto Agent Builder Examples

This directory contains examples demonstrating the **`AutoAgentBuilder`** harness — a builder that designs a roster of agents for a task instead of you writing them by hand.

## Examples

Start at the top — each one adds a single idea to the previous.

| Example | What it shows |
|---|---|
| [auto_agent_builder_simple.py](auto_agent_builder_simple.py) | Generate agent configurations and print them as JSON. Nothing is constructed or executed. |
| [auto_agent_builder_example.py](auto_agent_builder_example.py) | Generate a roster, build the `Agent` objects from it, and run them through a `SequentialWorkflow`. |
| [exact_agent_count.py](exact_agent_count.py) | `max_agents` versus `num_agents`, run side by side, so the difference between a ceiling and an exact count is visible in the output. |
| [concurrent_workflow.py](concurrent_workflow.py) | Fan the roster out with `ConcurrentWorkflow` when the agents are independent and have no reason to run in sequence. |
| [swarm_router_handoff.py](swarm_router_handoff.py) | Let the builder pick the team and `SwarmRouter` pick the architecture — swap `swarm_type` without touching the roster. |
| [save_and_reuse_roster.py](save_and_reuse_roster.py) | Design once, cache to JSON, and rebuild from disk on later runs. Also edits the configs before constructing. |

## Overview

Most multi-agent code starts with you hand-writing each agent: a name, a description, a system prompt, a model. `AutoAgentBuilder` moves that step to the model. You give it a task; it returns the roster.

A single builder agent is forced to call one function, `build_agents`, and answer with a list rather than prose. Because the provider enforces the tool schema, there is no markdown fence to strip and no JSON to fish out of a paragraph.

Each generated agent carries exactly four fields — the minimum needed to construct an `Agent`:

```
name, description, system_prompt, model_name
```

The builder designs the team and stops. It does not choose a multi-agent architecture and it does not execute anything, so you stay in control of what runs the roster.

## Two output shapes

`run()` returns whichever shape the builder was configured for:

```python
AutoAgentBuilder().run(task)                  # -> [Agent, Agent, ...]
AutoAgentBuilder(return_dict=True).run(task)  # -> [{...}, {...}, ...]
```

When you need one specific shape regardless of how the builder was configured, call it directly — these ignore `return_dict`:

```python
builder.build_agents(task)    # always Agent objects
builder.build_configs(task)   # always configuration dicts
```

## Controlling roster size

This is the setting most likely to surprise you.

| Parameter | Meaning |
|---|---|
| `max_agents` | A **ceiling**, not a target. The builder is instructed to prefer the smallest roster that covers the task, so it will routinely return fewer. |
| `num_agents` | An **exact** count. Overrides `max_agents` and the builder's prefer-fewer guidance. |

`max_agents=5` on a task that decomposes cleanly into three roles will give you three agents — that is the builder working correctly, not a failure. Use `num_agents=5` when you want exactly five:

```python
AutoAgentBuilder(num_agents=5, return_dict=True).run(task)
```

If the model still returns fewer than `num_agents`, a warning is logged rather than the shortfall passing silently. Agents cannot be fabricated for a task that does not support them.

See [exact_agent_count.py](exact_agent_count.py) for both settings run against the same task.

## Choosing an architecture

The builder answers *who is on the team*. It does not answer *how they run* — that is left to you on purpose, because the same roster is useful in different shapes:

| Structure | Use when | Example |
|---|---|---|
| `SequentialWorkflow` | Each agent builds on the previous one's output. | [auto_agent_builder_example.py](auto_agent_builder_example.py) |
| `ConcurrentWorkflow` | The agents are independent and cover separate ground. | [concurrent_workflow.py](concurrent_workflow.py) |
| `SwarmRouter` | You want to swap architectures without rewriting the wiring. | [swarm_router_handoff.py](swarm_router_handoff.py) |

Since the roster is just a list of agents, it drops into `MixtureOfAgents`, `HierarchicalSwarm`, `GroupChat`, or anything else that accepts one.

## One call per method

Every public method triggers a fresh LLM call, and the builder is not deterministic. Calling `build_configs()` and then `build_agents()` designs **two different rosters** — so what you printed may not be what you ran.

Generate once and reuse the result:

```python
configs = builder.build_configs(task)

agents = [
    Agent(
        agent_name=c["name"],
        agent_description=c["description"],
        system_prompt=c["system_prompt"],
        model_name=c["model_name"],
        max_loops=1,
    )
    for c in configs
]
```

For reproducible runs across processes, cache the configurations to disk and rebuild from the file — see [save_and_reuse_roster.py](save_and_reuse_roster.py). Because they are plain dicts, you can also edit them first: pin every agent to one model, tighten a system prompt, or drop an agent you disagree with.

## Requirements

The builder needs a model that supports function calling, plus a provider key:

```bash
export OPENAI_API_KEY=sk-...
```

Generated agents may name a different provider than the builder — the builder chooses a model per agent based on that agent's workload, and deliberately mixes tiers. If a roster references a provider you have no key for, either set that key or edit `model_name` on the configs before constructing.
