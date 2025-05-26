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
            llm=ChatOpenAI(model="gpt-4o"),
        )
        result = await agent.run()
        return result

    def run(self, task: str):
        return asyncio.run(self.browser_agent_test(task))


swarm = ConcurrentWorkflow(
    agents=[BrowserAgent() for _ in range(10)],
)

swarm.run(
    """
    Go to coinpost.jp and find the latest news about the crypto market.
    """
)
