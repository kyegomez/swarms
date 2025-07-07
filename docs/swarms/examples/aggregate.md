# Aggregate Multi-Agent Responses

The `aggregate` function allows you to run multiple agents concurrently on the same task and then synthesize their responses using an intelligent aggregator agent. This is useful for getting diverse perspectives on a problem and then combining them into a comprehensive analysis.

## Installation

You can get started by first installing swarms with the following command, or [click here for more detailed installation instructions](https://docs.swarms.world/en/latest/swarms/install/install/):

```bash
pip3 install -U swarms
``` 

## Environment Variables

```txt
WORKSPACE_DIR=""
OPENAI_API_KEY=""
ANTHROPIC_API_KEY=""
```

## How It Works

1. **Concurrent Execution**: All agents in the `workers` list run the same task simultaneously
2. **Response Collection**: Individual agent responses are collected into a conversation
3. **Intelligent Aggregation**: A specialized aggregator agent analyzes all responses and creates a comprehensive synthesis
4. **Formatted Output**: The final result is returned in the specified format

## Code Example

```python
from swarms.structs.agent import Agent
from swarms.structs.ma_blocks import aggregate


# Create specialized agents for different perspectives
agents = [
    Agent(
        agent_name="Sector-Financial-Analyst",
        agent_description="Senior financial analyst at BlackRock.",
        system_prompt="You are a financial analyst tasked with optimizing asset allocations for a $50B portfolio. Provide clear, quantitative recommendations for each sector.",
        max_loops=1,
        model_name="gpt-4o-mini",
        max_tokens=3000,
    ),
    Agent(
        agent_name="Sector-Risk-Analyst",
        agent_description="Expert risk management analyst.",
        system_prompt="You are a risk analyst responsible for advising on risk allocation within a $50B portfolio. Provide detailed insights on risk exposures for each sector.",
        max_loops=1,
        model_name="gpt-4o-mini",
        max_tokens=3000,
    ),
    Agent(
        agent_name="Tech-Sector-Analyst",
        agent_description="Technology sector analyst.",
        system_prompt="You are a tech sector analyst focused on capital and risk allocations. Provide data-backed insights for the tech sector.",
        max_loops=1,
        model_name="gpt-4o-mini",
        max_tokens=3000,
    ),
]

# Run the aggregate function
result = aggregate(
    workers=agents,
    task="What is the best sector to invest in?",
    type="all",  # Get complete conversation history
    aggregator_model_name="anthropic/claude-3-sonnet-20240229"
)

print(result)
```
