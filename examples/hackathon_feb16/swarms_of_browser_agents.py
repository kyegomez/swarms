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
    """Please navigate to chat.com and engage in a detailed technical discussion with ChatGPT about the following specific aspects of future high-energy physics:

1. The potential discoveries and physics reach of the Future Circular Collider (FCC) compared to the LHC
2. Theoretical predictions for supersymmetric particles and dark matter candidates at energy scales above 100 TeV
3. Novel detector technologies needed for future collider experiments, particularly for tracking and calorimetry
4. The role of machine learning and quantum computing in analyzing high-energy physics data
5. Challenges and proposed solutions for beam focusing and acceleration at extremely high energies

Please document the key insights and technical details from this discussion."""
)
