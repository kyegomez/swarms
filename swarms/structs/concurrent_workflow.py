import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Union

from swarms.structs.agent import Agent
from swarms.structs.base_swarm import BaseSwarm
from swarms.structs.conversation import Conversation
from swarms.utils.formatter import formatter
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.loguru_logger import initialize_logger

logger = initialize_logger(log_folder="concurrent_workflow")


class ConcurrentWorkflow(BaseSwarm):
    """
    Represents a concurrent workflow that executes multiple agents concurrently in a production-grade manner.
    Features include:
    - Caching for repeated prompts
    - Enhanced error handling and retries
    - Input validation

    Args:
        name (str): The name of the workflow. Defaults to "ConcurrentWorkflow".
        description (str): The description of the workflow. Defaults to "Execution of multiple agents concurrently".
        agents (List[Agent]): The list of agents to be executed concurrently. Defaults to an empty list.
        metadata_output_path (str): The path to save the metadata output. Defaults to "agent_metadata.json".
        auto_save (bool): Flag indicating whether to automatically save the metadata. Defaults to False.
        output_type (str): The type of output format. Defaults to "dict".
        max_loops (int): The maximum number of loops for each agent. Defaults to 1.
        return_str_on (bool): Flag indicating whether to return the output as a string. Defaults to False.
        auto_generate_prompts (bool): Flag indicating whether to auto-generate prompts for agents. Defaults to False.
        return_entire_history (bool): Flag indicating whether to return the entire conversation history. Defaults to False.
        cache_size (int): The size of the cache. Defaults to 100.
        max_retries (int): The maximum number of retry attempts. Defaults to 3.
        retry_delay (float): The delay between retry attempts in seconds. Defaults to 1.0.

    Raises:
        ValueError: If the list of agents is empty or if the description is empty.

    Attributes:
        name (str): The name of the workflow.
        description (str): The description of the workflow.
        agents (List[Agent]): The list of agents to be executed concurrently.
        metadata_output_path (str): The path to save the metadata output.
        auto_save (bool): Flag indicating whether to automatically save the metadata.
        output_type (str): The type of output format.
        max_loops (int): The maximum number of loops for each agent.
        return_str_on (bool): Flag indicating whether to return the output as a string.
        auto_generate_prompts (bool): Flag indicating whether to auto-generate prompts for agents.
        return_entire_history (bool): Flag indicating whether to return the entire conversation history.
        cache_size (int): The size of the cache.
        max_retries (int): The maximum number of retry attempts.
        retry_delay (float): The delay between retry attempts in seconds.
        _cache (dict): The cache for storing agent outputs.
    """

    def __init__(
        self,
        name: str = "ConcurrentWorkflow",
        description: str = "Execution of multiple agents concurrently",
        agents: List[Union[Agent, Callable]] = None,
        metadata_output_path: str = "agent_metadata.json",
        auto_save: bool = True,
        output_type: str = "dict-all-except-first",
        max_loops: int = 1,
        return_str_on: bool = False,
        auto_generate_prompts: bool = False,
        return_entire_history: bool = False,
        cache_size: int = 100,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        *args,
        **kwargs,
    ):
        agents = agents or []
        super().__init__(
            name=name,
            description=description,
            agents=agents,
            *args,
            **kwargs,
        )
        self.name = name
        self.description = description
        self.agents = agents
        self.metadata_output_path = metadata_output_path
        self.auto_save = auto_save
        self.max_loops = max_loops
        self.return_str_on = return_str_on
        self.auto_generate_prompts = auto_generate_prompts
        self.max_workers = os.cpu_count()
        self.output_type = output_type
        self.return_entire_history = return_entire_history
        self.tasks = []  # Initialize tasks list
        self.cache_size = cache_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._cache = {}

        self.reliability_check()
        self.conversation = Conversation()

    def reliability_check(self):
        try:
            formatter.print_panel(
                content=f"\n 🏷️ Name: {self.name}\n 📝 Description: {self.description}\n 🤖 Agents: {len(self.agents)}\n 🔄 Max Loops: {self.max_loops}\n ",
                title="⚙️ Concurrent Workflow Settings",
                style="bold blue",
            )
            formatter.print_panel(
                content="🔍 Starting reliability checks",
                title="🔒 Reliability Checks",
                style="bold blue",
            )

            if self.name is None:
                logger.error("❌ A name is required for the swarm")
                raise ValueError(
                    "❌ A name is required for the swarm"
                )

            if not self.agents or len(self.agents) <= 1:
                logger.error(
                    "❌ The list of agents must not be empty."
                )
                raise ValueError(
                    "❌ The list of agents must not be empty."
                )

            if not self.description:
                logger.error("❌ A description is required.")
                raise ValueError("❌ A description is required.")

            formatter.print_panel(
                content="✅ Reliability checks completed successfully",
                title="🎉 Reliability Checks",
                style="bold green",
            )

        except ValueError as e:
            logger.error(f"❌ Reliability check failed: {e}")
            raise
        except Exception as e:
            logger.error(
                f"💥 An unexpected error occurred during reliability checks: {e}"
            )
            raise

    def activate_auto_prompt_engineering(self):
        """
        Activates the auto-generate prompts feature for all agents in the workflow.

        Example:
            >>> workflow = ConcurrentWorkflow(agents=[Agent()])
            >>> workflow.activate_auto_prompt_engineering()
            >>> # All agents in the workflow will now auto-generate prompts.
        """
        if self.auto_generate_prompts is True:
            for agent in self.agents:
                agent.auto_generate_prompt = True

    @lru_cache(maxsize=100)
    def _cached_run(self, task: str, agent_id: int) -> Any:
        """Cached version of agent execution to avoid redundant computations"""
        return self.agents[agent_id].run(task=task)

    def _validate_input(self, task: str) -> bool:
        """Validate input task"""
        if not isinstance(task, str):
            raise ValueError("Task must be a string")
        if not task.strip():
            raise ValueError("Task cannot be empty")
        return True

    def _run_with_retry(
        self, agent: Agent, task: str, img: str = None
    ) -> Any:
        """Run agent with retry mechanism"""
        for attempt in range(self.max_retries):
            try:
                output = agent.run(task=task, img=img)
                self.conversation.add(agent.agent_name, output)
                return output
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(
                        f"Error running agent {agent.agent_name} after {self.max_retries} attempts: {e}"
                    )
                    raise
                logger.warning(
                    f"Attempt {attempt + 1} failed for agent {agent.agent_name}: {e}"
                )
                time.sleep(
                    self.retry_delay * (attempt + 1)
                )  # Exponential backoff

    def _process_agent(
        self, agent: Agent, task: str, img: str = None
    ) -> Any:
        """
        Process a single agent with caching and error handling.

        Args:
            agent: The agent to process
            task: Task to execute
            img: Optional image input

        Returns:
            The agent's output
        """
        try:
            # Fast path - check cache first
            cache_key = f"{task}_{agent.agent_name}"
            if cache_key in self._cache:
                output = self._cache[cache_key]
            else:
                # Slow path - run agent and update cache
                output = self._run_with_retry(agent, task, img)

                if len(self._cache) >= self.cache_size:
                    self._cache.pop(next(iter(self._cache)))

                self._cache[cache_key] = output

            return output
        except Exception as e:
            logger.error(
                f"Error running agent {agent.agent_name}: {e}"
            )
            raise

    def _run(
        self, task: str, img: str = None, *args, **kwargs
    ) -> Union[Dict[str, Any], str]:
        """
        Enhanced run method with parallel execution.
        """
        # Fast validation
        self._validate_input(task)
        self.conversation.add("User", task)

        try:
            # Parallel execution with optimized thread pool
            with ThreadPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                futures = [
                    executor.submit(
                        self._process_agent, agent, task, img
                    )
                    for agent in self.agents
                ]
                # Wait for all futures to complete
                for future in futures:
                    future.result()

        except Exception as e:
            logger.error(f"An error occurred during execution: {e}")
            raise e

        return history_output_formatter(
            self.conversation,
            type=self.output_type,
        )

    def run(
        self,
        task: Optional[str] = None,
        img: Optional[str] = None,
        *args,
        **kwargs,
    ) -> Any:
        """
        Executes the agent's run method with parallel execution.

        Args:
            task (Optional[str], optional): The task to be executed. Defaults to None.
            img (Optional[str], optional): The image to be processed. Defaults to None.
            *args: Additional positional arguments to be passed to the execution method.
            **kwargs: Additional keyword arguments to be passed to the execution method.

        Returns:
            Any: The result of the execution.

        Raises:
            ValueError: If task validation fails.
            Exception: If any other error occurs during execution.
        """
        if task is not None:
            self.tasks.append(task)

        try:
            outputs = self._run(task, img, *args, **kwargs)
            return outputs
        except Exception as e:
            logger.error(f"An error occurred during execution: {e}")
            raise e

    def run_batched(self, tasks: List[str]) -> Any:
        """
        Enhanced batched execution
        """
        if not tasks:
            raise ValueError("Tasks list cannot be empty")

        return [self.run(task) for task in tasks]

    def clear_cache(self):
        """Clear the task cache"""
        self._cache.clear()

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            "cache_size": len(self._cache),
            "max_cache_size": self.cache_size,
        }


# if __name__ == "__main__":
#     # Assuming you've already initialized some agents outside of this class
#     agents = [
#         Agent(
#             agent_name=f"Financial-Analysis-Agent-{i}",
#             system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
#             model_name="gpt-4o",
#             max_loops=1,
#         )
#         for i in range(3)  # Adjust number of agents as needed
#     ]

#     # Initialize the workflow with the list of agents
#     workflow = ConcurrentWorkflow(
#         agents=agents,
#         metadata_output_path="agent_metadata_4.json",
#         return_str_on=True,
#     )

#     # Define the task for all agents
#     task = "How can I establish a ROTH IRA to buy stocks and get a tax break? What are the criteria?"

#     # Run the workflow and save metadata
#     metadata = workflow.run(task)
#     print(metadata)
