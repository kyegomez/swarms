from swarms.structs.base_swarm import BaseSwarm
from typing import List
from swarms.structs.agent import Agent
from swarms.utils.loguru_logger import logger


class RoundRobinSwarm(BaseSwarm):
    """
    A swarm implementation that executes tasks in a round-robin fashion.

    Args:
        agents (List[Agent], optional): List of agents in the swarm. Defaults to None.
        verbose (bool, optional): Flag to enable verbose mode. Defaults to False.
        max_loops (int, optional): Maximum number of loops to run. Defaults to 1.
        callback (callable, optional): Callback function to be called after each loop. Defaults to None.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        agents (List[Agent]): List of agents in the swarm.
        verbose (bool): Flag to enable verbose mode.
        max_loops (int): Maximum number of loops to run.
        index (int): Current index of the agent being executed.

    Methods:
        run(task: str, *args, **kwargs) -> Any: Executes the given task on the agents in a round-robin fashion.

    """

    def __init__(
        self,
        agents: List[Agent] = None,
        verbose: bool = False,
        max_loops: int = 1,
        callback: callable = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.agents = agents
        self.verbose = verbose
        self.max_loops = max_loops
        self.callback = callback
        self.index = 0

    def run(self, task: str, *args, **kwargs):
        """
        Executes the given task on the agents in a round-robin fashion.

        Args:
            task (str): The task to be executed.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The result of the task execution.

        Raises:
            Exception: If an exception occurs during task execution.

        """
        try:
            result = task
            n = len(self.agents)
            logger.info(f"Running the task {task} on {n} agents.")
            for loop in range(self.max_loops):
                for _ in range(n):
                    current_agent = self.agents[self.index]
                    try:
                        logger.info(
                            f"Running Agent {current_agent.agent_name} on task {result}"
                        )
                        result = current_agent.run(result, *args, **kwargs)
                    except Exception as e:
                        logger.error(
                            f"Handling an exception for {current_agent.name}: {e}"
                        )
                        raise e
                    finally:
                        self.index = (
                            self.index + 1
                        ) % n  # Increment and wrap around the index
                if self.callback:
                    logger.info(
                        f"Calling the callback function for loop {loop}"
                    )
                    self.callback(loop, result)
            return result
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            return e
