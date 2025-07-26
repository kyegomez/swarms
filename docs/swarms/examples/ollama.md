# Agent with Ollama

- No API key needed
- Select your model_name like `ollama/llama2` follows [LiteLLM conventions](https://docs.litellm.ai/docs/providers/ollama)


```python
from swarms import Agent
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize the agent with ChromaDB memory
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    model_name="ollama/llama2",
    system_prompt="Agent system prompt here",
    agent_description="Agent performs financial analysis.",
)

# Run a query
agent.run("What are the components of a startup's stock incentive equity plan?")
```