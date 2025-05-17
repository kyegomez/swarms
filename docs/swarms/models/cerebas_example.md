# Using Cerebras LLaMA with Swarms

This guide demonstrates how to create and use an AI agent powered by the Cerebras LLaMA 3 70B model using the Swarms framework.

## Prerequisites

- Python 3.7+
- Swarms library installed (`pip install swarms`)

## Step-by-Step Guide

### 1. Import Required Module

```python
from swarms.structs.agent import Agent
```

This imports the `Agent` class from Swarms, which is the core component for creating AI agents.

### 2. Create an Agent Instance

```python
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    agent_description="Personal finance advisor agent",
    max_loops=4,
    model_name="cerebras/llama3-70b-instruct",
    dynamic_temperature_enabled=True,
    interactive=False,
    output_type="all",
)
```

Let's break down each parameter:

- `agent_name`: A descriptive name for your agent (here, "Financial-Analysis-Agent")

- `agent_description`: A brief description of the agent's purpose

- `max_loops`: Maximum number of interaction loops the agent can perform (set to 4)

- `model_name`: Specifies the Cerebras LLaMA 3 70B model to use

- `dynamic_temperature_enabled`: Enables dynamic adjustment of temperature for varied responses

- `interactive`: When False, runs without requiring user interaction

- `output_type`: Set to "all" to return complete response information

### 3. Run the Agent

```python
agent.run("Conduct an analysis of the best real undervalued ETFs")
```

This command:

1. Activates the agent

2. Processes the given prompt about ETF analysis

3. Returns the analysis based on the model's knowledge

## Notes

- The Cerebras LLaMA 3 70B model is a powerful language model suitable for complex analysis tasks

- The agent can be customized further with additional parameters

- The `max_loops=4` setting prevents infinite loops while allowing sufficient processing depth

- Setting `interactive=False` makes the agent run autonomously without user intervention

## Example Output

The agent will provide a detailed analysis of undervalued ETFs, including:

- Market analysis

- Performance metrics

- Risk assessment

- Investment recommendations

Note: Actual output will vary based on current market conditions and the model's training data.