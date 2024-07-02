import threading
from queue import Queue
from typing import List
from swarms.structs.agent import Agent
from swarms.utils.loguru_logger import logger
from dotenv import load_dotenv
import os
from swarms.models.popular_llms import OpenAIChat


class ConcurrentWorkflow:
    def __init__(self, agents: List[Agent], max_loops: int):
        """
        Initializes the ConcurrentWorkflow with the given parameters.

        Args:
            agents (List[Agent]): The list of agents to initialize.
            max_loops (int): The maximum number of loops each agent can run.
        """
        self.max_loops = max_loops
        self.agents = agents
        self.num_agents = len(agents)
        self.output_queue = Queue()

    def run_agent(self, agent: Agent, task: str) -> None:
        """
        Runs a given agent on the specified task once.

        Args:
            agent (Agent): The agent to run.
            task (str): The task for the agent to execute.
        """
        try:
            logger.info(f"Running agent {agent} on task '{task}'")
            result = agent.run(task)
            logger.info(
                f"Agent {agent} completed task with result: {result}"
            )

            if result is None:
                raise ValueError("Received None as result")

            self.output_queue.put(result)
        except Exception as e:
            logger.error(f"Error running agent {agent}: {e}")
            self.output_queue.put(f"Error: {e}")

    def process_agent_outputs(self, task: str) -> None:
        """
        Processes outputs from agents and conditionally sends them to other agents.

        Args:
            task (str): The task for the agents to execute.
        """
        while not self.output_queue.empty():
            result = self.output_queue.get()
            if isinstance(result, str) and result.startswith("Error:"):
                logger.error(result)
            else:
                logger.info(f"Processing result: {result}")
                for next_agent in self.agents:
                    self.run_agent(next_agent, task)

    def run(self, task: str) -> str:
        """
        Runs a list of agents concurrently on the same task using threads.

        Args:
            task (str): The task for the agents to execute.

        Returns:
            str: The final result of the concurrent execution.
        """
        threads = []

        try:
            for agent in self.agents:
                thread = threading.Thread(
                    target=self.run_agent, args=(agent, task)
                )
                thread.start()
                threads.append(thread)

            for thread in threads:
                thread.join()

            # self.process_agent_outputs(task)
        except Exception as e:
            logger.error(f"Error in concurrent workflow: {e}")

        return None


# Load the environment variables
load_dotenv()

# Get the API key from the environment
api_key = os.environ.get("OPENAI_API_KEY")

# Initialize the language model (assuming this should be done outside the class and passed to it)
llm = OpenAIChat(temperature=0.5, openai_api_key=api_key, max_tokens=4000)

# Initialize agents
agents = [
    Agent(llm=llm, max_loops=1, autosave=True, dashboard=True)
    for _ in range(1)
]

# Task to be executed by each agent
task = "Generate a 10,000 word blog on health and wellness."

# Initialize and run the ConcurrentWorkflow
workflow = ConcurrentWorkflow(agents=agents, max_loops=1)
result = workflow.run(task)
logger.info(f"Final result: {result}")
