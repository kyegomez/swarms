import threading
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Any

from swarms.utils.logger import logger
from swarms.structs.agent import Agent
from swarms.structs.base_workflow import BaseWorkflow
from swarms import OpenAIChat
import os


@dataclass
class ConcurrentWorkflow(BaseWorkflow):
    """
    ConcurrentWorkflow class for running a set of tasks concurrently using N number of autonomous agents.

    Args:
        max_workers (int): The maximum number of workers to use for the threading.Thread.
        autosave (bool): Whether to save the state of the workflow to a file. Default is False.
        saved_state_filepath (str): The filepath to save the state of the workflow to. Default is "runs/concurrent_workflow.json".
        print_results (bool): Whether to print the results of each task. Default is False.
        return_results (bool): Whether to return the results of each task. Default is False.
        use_processes (bool): Whether to use processes instead of threads. Default is False.

    Examples:
    >>> from swarms.models import OpenAIChat
    >>> from swarms.structs import ConcurrentWorkflow
    >>> llm = OpenAIChat(openai_api_key="")
    >>> workflow = ConcurrentWorkflow(max_workers=5, agents=[llm])
    >>> workflow.run()
    """

    max_loops: int = 1
    max_workers: int = 5
    autosave: bool = False
    agents: List[Agent] = field(default_factory=list)
    saved_state_filepath: Optional[str] = "runs/concurrent_workflow.json"
    print_results: bool = True  # Modified: Set print_results to True
    return_results: bool = False
    stopping_condition: Optional[Callable] = None

    def run(
        self, task: Optional[str] = None, *args, **kwargs
    ) -> Optional[List[Any]]:
        """
        Executes the tasks in parallel using multiple threads.

        Args:
            task (Optional[str]): A task description if applicable.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Optional[List[Any]]: A list of the results of each task, if return_results is True. Otherwise, returns None.
        """
        loop = 0
        results = []

        while loop < self.max_loops:
            if not self.agents:
                logger.warning("No agents found in the workflow.")
                break

            threads = [
                threading.Thread(
                    target=self.execute_agent, args=(agent, task)
                )
                for agent in self.agents
            ]

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()

            if self.return_results:
                results.extend(
                    [
                        thread.result
                        for thread in threads
                        if hasattr(thread, "result")
                    ]
                )

            loop += 1

            if self.stopping_condition and self.stopping_condition(
                results
            ):
                break

        return results if self.return_results else None

    def list_agents(self):
        """Prints a list of the agents in the workflow."""
        for agent in self.agents:
            logger.info(agent)

    def save(self):
        """Saves the state of the workflow to a file."""
        self.save_state(self.saved_state_filepath)

    def execute_agent(
        self, agent: Agent, task: Optional[str] = None, *args, **kwargs
    ):
        try:
            result = agent.run(task, *args, **kwargs)
            if self.print_results:
                logger.info(f"Agent {agent}: {result}")
            if self.return_results:
                return result
        except Exception as e:
            logger.error(f"Agent {agent} generated an exception: {e}")


api_key = os.environ["OPENAI_API_KEY"]

# Model
swarm = ConcurrentWorkflow(
    agents=[
        Agent(
            llm=OpenAIChat(
                openai_api_key=api_key,
                max_tokens=4000,
            ),
            max_loops=4,
            dashboard=False,
        )
    ],
)


# Run the workflow
swarm.run(
    "Generate a report on the top 3 biggest expenses for small businesses and how businesses can save 20%"
)
