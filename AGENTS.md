# AGENTS.md

Guidance for AI coding agents (Codex, Cursor, Copilot, Jules, Claude Code and others) working with **Swarms**, a multi-agent orchestration framework for Python.

## Quickstart

```bash
pip install -U swarms
export OPENAI_API_KEY="sk-..."   # or any provider key
```

```python
from swarms import Agent

agent = Agent(agent_name="Analyst", model_name="gpt-5.4", max_loops=1)
print(agent.run("Summarise the current state of LLM research."))
```

## Where to look

- **[`CLAUDE.md`](./CLAUDE.md)**: the full guide. `Agent` parameters, every multi-agent structure, tools, streaming, memory, MCP, and how to choose between them. Read it before writing Swarms code.
- **[`SKILL.md`](./SKILL.md)**: a condensed version packaged as an agent skill.
- **[`examples/`](./examples)**: hundreds of runnable examples. One is probably close to what you need.
- **Docs MCP server**: `https://docs.swarms.world/mcp`, no key. Search the current docs instead of relying on training data; names changed across v12 to v16.
- **Marketplace MCP server**: `https://swarms.world/mcp`. Listing tools needs no key; calling them needs a Swarms API key as `Authorization: Bearer <key>`.
- **Plain-text docs**: [llms.txt](https://docs.swarms.world/llms.txt) and [llms-full.txt](https://docs.swarms.world/llms-full.txt).

## Rules

1. Import from the top level: `from swarms import Agent`, not `from swarms.structs.agent import Agent`.
2. Give every agent a unique `agent_name`; memory and routing key on it.
3. Prefer an explicit `max_loops=N`. Use `"auto"` only for open-ended work.
4. Don't set `streaming_on=True` and `streaming_callback` on the same agent.

## Contributing to this repository

- Commit, PR and issue titles use WARP: `[TYPE][Function/FileName][Short description]`.
- Disclose AI authorship: a co-author trailer on commits and a "Generated with" line naming the agent and model on PRs and issues.
- Format with `black --line-length 70` and lint with `ruff check .`.
- User-visible changes need a matching PR in [swarms-framework-docs](https://github.com/The-Swarm-Corporation/swarms-framework-docs).

`CLAUDE.md` has the full contribution rules.
