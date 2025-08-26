import asyncio

from browser_use import Agent
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from swarms import ConcurrentWorkflow

load_dotenv()


class BrowserAgent:
    def __init__(self, agent_name: str = "BrowserAgent"):
        """
        Initialize a BrowserAgent with a given name.

        Args:
            agent_name (str): The name of the browser agent.
        """
        self.agent_name = agent_name

    async def browser_agent_test(self, task: str):
        """
        Asynchronously run the browser agent on a given task.

        Args:
            task (str): The task prompt for the agent.

        Returns:
            Any: The result of the agent's run method.
        """
        agent = Agent(
            task=task,
            llm=ChatOpenAI(model="gpt-4.1"),
        )
        result = await agent.run()
        return result

    def run(self, task: str):
        """
        Run the browser agent synchronously on a given task.

        Args:
            task (str): The task prompt for the agent.

        Returns:
            Any: The result of the agent's run method.
        """
        return asyncio.run(self.browser_agent_test(task))


swarm = ConcurrentWorkflow(
    agents=[BrowserAgent() for _ in range(10)],
)

swarm.run(
    """Please navigate to https://www.coingecko.com and identify the best performing cryptocurrency coin over the past 24 hours. 

Your task is to:
1. Go to the main page of CoinGecko.
2. Locate the list of coins and their performance metrics.
3. Determine which coin has the highest positive percentage change in price over the last 24 hours.
4. Report the name, symbol, and 24h percentage gain of this top-performing coin.
5. Briefly describe any notable trends or observations about this coin's recent performance.

Please provide your findings in a clear and concise summary."""
)
