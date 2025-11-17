# Agent with DeepSeek

- Add your `DEEPSEEK_API_KEY` in the `.env` file

- Select your model_name like `deepseek/deepseek-chat` follows [LiteLLM conventions](https://docs.litellm.ai/docs/providers/deepseek)

- Execute your agent!


```python
from swarms import Agent
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize the agent with ChromaDB memory
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    model_name="deepseek/deepseek-chat",
    system_prompt="Agent system prompt here",
    agent_description="Agent performs financial analysis.",
)

# Run a query
agent.run("What are the components of a startup's stock incentive equity plan?")
```

## R1 

This is a simple example of how to use the DeepSeek Reasoner model otherwise known as R1.

```python

import os
from swarms import Agent
from dotenv import load_dotenv

load_dotenv()

# Initialize the agent with ChromaDB memory
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    model_name="deepseek/deepseek-reasoner",
    system_prompt="Agent system prompt here",
    agent_description="Agent performs financial analysis.",
)

# Run a query
agent.run("What are the components of a startup's stock incentive equity plan?")
```