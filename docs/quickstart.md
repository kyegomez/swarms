# Quickstart

This quickstart creates a Swarms agent, sends it a task, and prints the result.
It is the shortest path from a fresh environment to a working local agent.

## Install

```bash
pip install swarms
```

Set an API key for the model provider you want to use. For OpenAI-compatible
models, export `OPENAI_API_KEY` before running the example.

```bash
export OPENAI_API_KEY="your-api-key"
```

## Create An Agent

```python
from swarms import Agent

agent = Agent(
    agent_name="Research-Agent",
    system_prompt="You are a concise research assistant.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

response = agent.run("List three practical uses for multi-agent workflows.")
print(response)
```

## Next Steps

- Configure environment variables with the [environment setup guide](swarms/install/env.md).
- Explore model choices in the [model providers guide](swarms/examples/model_providers.md).
- Add tools with the [tools documentation](swarms/tools/main.md).
- Move from one agent to teams with the [multi-agent architecture guide](swarms/structs/index.md).
