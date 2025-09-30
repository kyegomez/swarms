# Swarms x Browser Use

- Import required modules

- Configure your agent first by making a new class

- Set your api keys for your model provider in the `.env` file such as `OPENAI_API_KEY="sk-"`

- Conigure your `ConcurrentWorkflow`

## Install

```bash
pip install swarms browser-use langchain-openai
```
--------


## Main
```python
import asyncio

from browser_use import Agent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from swarms import ConcurrentWorkflow

load_dotenv()


class BrowserAgent:
    def __init__(self, agent_name: str = "BrowserAgent"):
        self.agent_name = agent_name

    async def browser_agent_test(self, task: str):
        agent = Agent(
            task=task,
            llm=ChatOpenAI(model="gpt-4.1"),
        )
        result = await agent.run()
        return result

    def run(self, task: str):
        return asyncio.run(self.browser_agent_test(task))


swarm = ConcurrentWorkflow(
    agents=[BrowserAgent() for _ in range(3)],
)

swarm.run(
    """
    Go to pump.fun.
    
    2. Make an account: use email: "test@test.com" and password: "test1234"
    
    3. Make a coin called and give it a cool description and etc. Fill in the form
    
    4. Sit back and watch the coin grow in value.
    
    """
)

```