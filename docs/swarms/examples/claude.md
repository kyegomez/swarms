# Agent with Anthropic/Claude

- Get their api keys and put it in the `.env`

- Select your model_name like `claude-3-sonnet-20240229` follows LiteLLM conventions


```python
from swarms import Agent
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize the agent with ChromaDB memory
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    model_name="claude-3-sonnet-20240229",
    system_prompt="Agent system prompt here",
    agent_description="Agent performs financial analysis.",
)

# Run a query
agent.run("What are the components of a startup's stock incentive equity plan?")
```