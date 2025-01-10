# Agent with Cohere

- Add your `COHERE_API_KEY` in the `.env` file
- Select your model_name like `command-r` follows LiteLLM conventions


```python
from swarms import Agent
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize the agent with ChromaDB memory
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    model_name="command-r",
    system_prompt="Agent system prompt here",
    agent_description="Agent performs financial analysis.",
    llm=model,
)

# Run a query
agent.run("What are the components of a startup's stock incentive equity plan?")
```