import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Union

from tqdm import tqdm

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
    - Interactive model support
    - Caching for repeated prompts
    - Optional progress tracking
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
        interactive (bool): Flag indicating whether to enable interactive mode. Defaults to False.
        cache_size (int): The size of the cache. Defaults to 100.
        max_retries (int): The maximum number of retry attempts. Defaults to 3.
        retry_delay (float): The delay between retry attempts in seconds. Defaults to 1.0.
        show_progress (bool): Flag indicating whether to show progress. Defaults to False.

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
        interactive (bool): Flag indicating whether to enable interactive mode.
        cache_size (int): The size of the cache.
        max_retries (int): The maximum number of retry attempts.
        retry_delay (float): The delay between retry attempts in seconds.
        show_progress (bool): Flag indicating whether to show progress.
        _cache (dict): The cache for storing agent outputs.
        _progress_bar (tqdm): The progress bar for tracking execution.
    """

    def __init__(
        self,
        name: str = "ConcurrentWorkflow",
        description: str = "Execution of multiple agents concurrently",
        agents: List[Union[Agent, Callable]] = [],
        metadata_output_path: str = "agent_metadata.json",
        auto_save: bool = True,
        output_type: str = "dict-all-except-first",
        max_loops: int = 1,
        return_str_on: bool = False,
        auto_generate_prompts: bool = False,
        return_entire_history: bool = False,
        interactive: bool = False,
        cache_size: int = 100,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        show_progress: bool = False,
        *args,
        **kwargs,
    ):
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
        self.interactive = interactive
        self.cache_size = cache_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.show_progress = show_progress
        self._cache = {}
        self._progress_bar = None

        self.reliability_check()
        self.conversation = Conversation()

    def disable_agent_prints(self):
        for agent in self.agents:
            agent.no_print = False

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

    def enable_progress_bar(self):
        """Enable progress bar display"""
        self.show_progress = True

    def disable_progress_bar(self):
        """Disable progress bar display"""
        if self._progress_bar:
            self._progress_bar.close()
            self._progress_bar = None
        self.show_progress = False

    def _create_progress_bar(self, total: int):
        """Create a progress bar for tracking execution"""
        if self.show_progress:
            try:
                self._progress_bar = tqdm(
                    total=total,
                    desc="Processing tasks",
                    unit="task",
                    disable=not self.show_progress,
                    ncols=100,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                )
            except Exception as e:
                logger.warning(f"Failed to create progress bar: {e}")
                self.show_progress = False
                self._progress_bar = None
        return self._progress_bar

    def _update_progress(self, increment: int = 1):
        """Update the progress bar"""
        if self._progress_bar and self.show_progress:
            try:
                self._progress_bar.update(increment)
            except Exception as e:
                logger.warning(f"Failed to update progress bar: {e}")
                self.disable_progress_bar()

    def _validate_input(self, task: str) -> bool:
        """Validate input task"""
        if not isinstance(task, str):
            raise ValueError("Task must be a string")
        if not task.strip():
            raise ValueError("Task cannot be empty")
        return True

    def _handle_interactive(self, task: str) -> str:
        """Handle interactive mode for task input"""
        if self.interactive:
            from swarms.utils.formatter import formatter

            # Display current task in a panel
            formatter.print_panel(
                content=f"Current task: {task}",
                title="Task Status",
                style="bold blue",
            )

            # Get user input with formatted prompt
            formatter.print_panel(
                content="Do you want to modify this task? (y/n/q to quit): ",
                title="User Input",
                style="bold green",
            )
            response = input().lower()

            if response == "q":
                return None
            elif response == "y":
                formatter.print_panel(
                    content="Enter new task: ",
                    title="New Task Input",
                    style="bold yellow",
                )
                new_task = input()
                return new_task
        return task

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

    def _run(
        self, task: str, img: str = None, *args, **kwargs
    ) -> Union[Dict[str, Any], str]:
        """
        Enhanced run method with caching, progress tracking, and better error handling
        """

        # Validate and potentially modify task
        self._validate_input(task)
        task = self._handle_interactive(task)

        # Add task to conversation
        self.conversation.add("User", task)

        # Create progress bar if enabled
        if self.show_progress:
            self._create_progress_bar(len(self.agents))

        def run_agent(
            agent: Agent, task: str, img: str = None
        ) -> Any:
            try:
                # Check cache first
                cache_key = f"{task}_{agent.agent_name}"
                if cache_key in self._cache:
                    output = self._cache[cache_key]
                else:
                    output = self._run_with_retry(agent, task, img)
                    # Update cache
                    if len(self._cache) >= self.cache_size:
                        self._cache.pop(next(iter(self._cache)))
                    self._cache[cache_key] = output

                self._update_progress()
                return output
            except Exception as e:
                logger.error(
                    f"Error running agent {agent.agent_name}: {e}"
                )
                self._update_progress()
                raise

        try:
            with ThreadPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                list(
                    executor.map(
                        lambda agent: run_agent(agent, task),
                        self.agents,
                    )
                )
        finally:
            if self._progress_bar and self.show_progress:
                try:
                    self._progress_bar.close()
                except Exception as e:
                    logger.warning(
                        f"Failed to close progress bar: {e}"
                    )
                finally:
                    self._progress_bar = None

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
        Executes the agent's run method on a specified device with optional interactive mode.

        This method attempts to execute the agent's run method on a specified device, either CPU or GPU.
        It supports both standard execution and interactive mode where users can modify tasks and continue
        the workflow interactively.

        Args:
            task (Optional[str], optional): The task to be executed. Defaults to None.
            img (Optional[str], optional): The image to be processed. Defaults to None.
            is_last (bool, optional): Indicates if this is the last task. Defaults to False.
            device (str, optional): The device to use for execution. Defaults to "cpu".
            device_id (int, optional): The ID of the GPU to use if device is set to "gpu". Defaults to 0.
            all_cores (bool, optional): If True, uses all available CPU cores. Defaults to True.
            all_gpus (bool, optional): If True, uses all available GPUS. Defaults to True.
            *args: Additional positional arguments to be passed to the execution method.
            **kwargs: Additional keyword arguments to be passed to the execution method.

        Returns:
            Any: The result of the execution.

        Raises:
            ValueError: If an invalid device is specified.
            Exception: If any other error occurs during execution.
        """
        if task is not None:
            self.tasks.append(task)

        try:
            # Handle interactive mode
            if self.interactive:
                current_task = task
                loop_count = 0

                while loop_count < self.max_loops:
                    if (
                        self.max_loops is not None
                        and loop_count >= self.max_loops
                    ):
                        formatter.print_panel(
                            content=f"Maximum number of loops ({self.max_loops}) reached.",
                            title="Session Complete",
                            style="bold red",
                        )
                        break

                    if current_task is None:
                        formatter.print_panel(
                            content="Enter your task (or 'q' to quit): ",
                            title="Task Input",
                            style="bold blue",
                        )
                        current_task = input()
                        if current_task.lower() == "q":
                            break

                    # Run the workflow with the current task
                    try:
                        outputs = self._run(
                            current_task, img, *args, **kwargs
                        )
                        formatter.print_panel(
                            content=str(outputs),
                            title="Workflow Result",
                            style="bold green",
                        )
                    except Exception as e:
                        formatter.print_panel(
                            content=f"Error: {str(e)}",
                            title="Error",
                            style="bold red",
                        )

                    # Ask if user wants to continue
                    formatter.print_panel(
                        content="Do you want to continue with a new task? (y/n): ",
                        title="Continue Session",
                        style="bold yellow",
                    )
                    if input().lower() != "y":
                        break

                    current_task = None
                    loop_count += 1

                formatter.print_panel(
                    content="Interactive session ended.",
                    title="Session Complete",
                    style="bold blue",
                )
                return outputs
            else:
                # Standard non-interactive execution
                outputs = self._run(task, img, *args, **kwargs)
                return outputs

        except ValueError as e:
            logger.error(f"Invalid device specified: {e}")
            raise e
        except Exception as e:
            logger.error(f"An error occurred during execution: {e}")
            raise e

    def run_batched(self, tasks: List[str]) -> Any:
        """
        Enhanced batched execution with progress tracking
        """
        if not tasks:
            raise ValueError("Tasks list cannot be empty")

        results = []

        # Create progress bar if enabled
        if self.show_progress:
            self._create_progress_bar(len(tasks))

        try:
            for task in tasks:
                result = self.run(task)
                results.append(result)
                self._update_progress()
        finally:
            if self._progress_bar and self.show_progress:
                try:
                    self._progress_bar.close()
                except Exception as e:
                    logger.warning(
                        f"Failed to close progress bar: {e}"
                    )
                finally:
                    self._progress_bar = None

        return results

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
