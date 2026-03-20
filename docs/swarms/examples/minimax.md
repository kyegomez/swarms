# Agent with MiniMax

[MiniMax](https://www.minimax.io/) provides powerful language models with 204K context window and strong multilingual capabilities through an OpenAI-compatible API.

## Setup

1. Get your API key from the [MiniMax platform](https://platform.minimax.chat/)
2. Add `MINIMAX_API_KEY` to your `.env` file

```bash
MINIMAX_API_KEY=your_minimax_api_key
```

## Available Models

| Model | Description |
|-------|-------------|
| `MiniMax-M2.7` | Latest flagship model with 204K context, strong reasoning and multilingual support |
| `MiniMax-M2.5` | Balanced performance model with 204K context |
| `MiniMax-M2.5-highspeed` | High-speed variant optimized for fast inference with 204K context |

## Basic Usage

Use the `minimax/` prefix with the model name to automatically route to the MiniMax API:

```python
from swarms import Agent
from dotenv import load_dotenv

load_dotenv()

agent = Agent(
    agent_name="MiniMax-Agent",
    model_name="minimax/MiniMax-M2.7",
    system_prompt="You are a helpful assistant.",
    agent_description="Agent powered by MiniMax M2.7.",
    max_loops=1,
)

response = agent.run("Explain the key differences between supervised and unsupervised learning.")
print(response)
```

## High-Speed Model

For latency-sensitive applications, use the high-speed variant:

```python
from swarms import Agent
from dotenv import load_dotenv

load_dotenv()

agent = Agent(
    agent_name="Fast-MiniMax-Agent",
    model_name="minimax/MiniMax-M2.5-highspeed",
    system_prompt="You are a fast and concise assistant.",
    agent_description="Agent using MiniMax high-speed model.",
    max_loops=1,
    temperature=0.3,
)

response = agent.run("Summarize the benefits of microservices architecture.")
print(response)
```

## Multi-Agent Workflow

Combine MiniMax with other providers in a multi-agent setup:

```python
from swarms import Agent, ConcurrentWorkflow
from dotenv import load_dotenv

load_dotenv()

# MiniMax agent for long-context analysis
analysis_agent = Agent(
    agent_name="Analysis-Agent",
    model_name="minimax/MiniMax-M2.7",
    system_prompt="You are an expert analyst.",
    max_loops=1,
)

# Another agent for summarization
summary_agent = Agent(
    agent_name="Summary-Agent",
    model_name="minimax/MiniMax-M2.5-highspeed",
    system_prompt="You are an expert summarizer.",
    max_loops=1,
)

workflow = ConcurrentWorkflow(
    name="Analysis-Summary-Workflow",
    agents=[analysis_agent, summary_agent],
)
```

## Notes

- **Temperature**: MiniMax supports temperature values in the range `[0, 1.0]`. Values above 1.0 are automatically clamped.
- **Context Window**: All MiniMax models support up to 204K tokens of context.
- **API Compatibility**: MiniMax uses an OpenAI-compatible API, so all standard features (streaming, function calling, etc.) are supported.
