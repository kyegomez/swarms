
# Basic Usage Guide

## Getting Started

This guide demonstrates how to use the basic features of the Swarms framework.

### Basic Agent Example

```python
from swarms.structs.agent import Agent

# Initialize agent
agent = Agent(
    agent_name="Basic-Example-Agent",
    agent_description="A simple example agent",
    system_prompt="You are a helpful assistant.",
    model_name="gpt-4",
)

# Run the agent
response = agent.run("What is 2+2?")
print(f"Agent response: {response}")
```
