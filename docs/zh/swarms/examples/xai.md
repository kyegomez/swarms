# Agent with XAI

- Add your `XAI_API_KEY` in the `.env` file

- Select your model_name like `xai/grok-beta` follows [LiteLLM conventions](https://docs.litellm.ai/docs/providers/xai)

- Execute your agent!


```python
from swarms import Agent
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize the agent with ChromaDB memory
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    model_name="xai/grok-beta",
    system_prompt="Agent system prompt here",
    agent_description="Agent performs financial analysis.",
)

# Run a query
agent.run("What are the components of a startup's stock incentive equity plan?")
```