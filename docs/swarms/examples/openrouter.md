# Agent with OpenRouter

- Add your `OPENROUTER_API_KEY` in the `.env` file

- Select your model_name like `openrouter/google/palm-2-chat-bison` follows [LiteLLM conventions](https://docs.litellm.ai/docs/providers/openrouter)

- Execute your agent!


```python
from swarms import Agent
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize the agent with ChromaDB memory
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    model_name="openrouter/google/palm-2-chat-bison",
    system_prompt="Agent system prompt here",
    agent_description="Agent performs financial analysis.",
)

# Run a query
agent.run("What are the components of a startup's stock incentive equity plan?")
```