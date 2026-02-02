import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional, Union

from loguru import logger as loguru_logger
from swarms.prompts.multi_agent_collab_prompt import (
    MULTI_AGENT_COLLAB_PROMPT,
)
from swarms.structs.agent import Agent
from swarms.structs.agent_rearrange import AgentRearrange
from swarms.utils.loguru_logger import initialize_logger
from swarms.utils.output_types import OutputType
from swarms.utils.swarm_autosave import get_swarm_workspace_dir
from swarms.utils.workspace_utils import get_workspace_dir

logger = initialize_logger(log_folder="sequential_workflow")


class SequentialWorkflow:
    """
    Orchestrates the execution of a sequence of agents in a defined workflow.

    This class enables the construction and execution of a workflow where multiple agents
    (or callables) are executed in a specified order, passing tasks and optional data
    through the chain. It supports both synchronous and asynchronous execution, as well as
    batched and concurrent task processing.

    Attributes:
        id (str): Unique identifier for the workflow instance.
        name (str): Human-readable name for the workflow.
        description (str): Description of the workflow's purpose.
        agents (List[Union[Agent, Callable]]): List of agents or callables to execute in sequence.
        max_loops (int): Maximum number of times to execute the workflow.
        output_type (OutputType): Format of the output from the workflow.
        shared_memory_system (callable): Optional callable for managing shared memory between agents.
        multi_agent_collab_prompt (bool): Whether to append a collaborative prompt to each agent.
        flow (str): String representation of the agent execution order.
        agent_rearrange (AgentRearrange): Internal helper for managing agent execution.

    Raises:
        ValueError: If the agents list is None or empty, or if max_loops is set to 0.
    """

    def __init__(
        self,
        id: str = "sequential_workflow",
        name: str = "SequentialWorkflow",
        description: str = "Sequential Workflow, where agents are executed in a sequence.",
        agents: List[Union[Agent, Callable]] = None,
        max_loops: int = 1,
        output_type: OutputType = "dict",
        shared_memory_system: callable = None,
        multi_agent_collab_prompt: bool = False,
        team_awareness: bool = False,
        autosave: bool = True,
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initialize a SequentialWorkflow instance.

        Args:
            id (str, optional): Unique identifier for the workflow. Defaults to "sequential_workflow".
            name (str, optional): Name of the workflow. Defaults to "SequentialWorkflow".
            description (str, optional): Description of the workflow. Defaults to a standard description.
            agents (List[Union[Agent, Callable]], optional): List of agents or callables to execute in sequence.
            max_loops (int, optional): Maximum number of times to execute the workflow. Defaults to 1.
            output_type (OutputType, optional): Output format for the workflow. Defaults to "dict".
            shared_memory_system (callable, optional): Callable for shared memory management. Defaults to None.
            multi_agent_collab_prompt (bool, optional): If True, appends a collaborative prompt to each agent.
            autosave (bool, optional): Whether to enable autosaving of conversation history. Defaults to False.
            verbose (bool, optional): Whether to enable verbose logging. Defaults to False.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Raises:
            ValueError: If the agents list is None or empty, or if max_loops is set to 0.
        """
        self.id = id
        self.name = name
        self.description = description
        self.agents = agents
        self.max_loops = max_loops
        self.output_type = output_type
        self.shared_memory_system = shared_memory_system
        self.multi_agent_collab_prompt = multi_agent_collab_prompt
        self.team_awareness = team_awareness
        self.autosave = autosave
        self.verbose = verbose
        self.swarm_workspace_dir = None

        self.reliability_check()
        self.flow = self.sequential_flow()

        self.agent_rearrange = AgentRearrange(
            name=self.name,
            description=self.description,
            agents=self.agents,
            flow=self.flow,
            max_loops=self.max_loops,
            output_type=self.output_type,
            team_awareness=self.team_awareness,
        )

        # Setup autosave workspace if enabled
        if self.autosave:
            self._setup_autosave()

    def reliability_check(self):
        """
        Validates the workflow configuration and prepares agents for execution.

        Raises:
            ValueError: If the agents list is None or empty, or if max_loops is set to 0.
        """
        if self.agents is None or len(self.agents) == 0:
            raise ValueError("Agents list cannot be None or empty")

        if self.max_loops == 0:
            raise ValueError("max_loops cannot be 0")

        if self.multi_agent_collab_prompt is True:
            for agent in self.agents:
                if hasattr(agent, "system_prompt"):
                    if agent.system_prompt is None:
                        agent.system_prompt = ""
                    agent.system_prompt += MULTI_AGENT_COLLAB_PROMPT
                else:
                    logger.warning(
                        f"Agent {getattr(agent, 'name', str(agent))} does not have a 'system_prompt' attribute."
                    )

        logger.info(
            f"Sequential Workflow Name: {self.name} is ready to run."
        )

    def sequential_flow(self):
        """
        Constructs a string representation of the agent execution order.

        Returns:
            str: A string showing the order of agent execution (e.g., "AgentA -> AgentB -> AgentC").
                 Returns an empty string if no valid agent names are found.
        """
        if self.agents:
            agent_names = []
            for agent in self.agents:
                try:
                    # Try to get agent_name, fallback to name if not available
                    agent_name = (
                        getattr(agent, "agent_name", None)
                        or agent.name
                    )
                    agent_names.append(agent_name)
                except AttributeError:
                    logger.warning(
                        f"Could not get name for agent {agent}"
                    )
                    continue

            if agent_names:
                flow = " -> ".join(agent_names)
            else:
                flow = ""
                logger.warning(
                    "No valid agent names found to create flow"
                )
        else:
            flow = ""
            logger.warning("No agents provided to create flow")

        return flow

    def run(
        self,
        task: str,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
        *args,
        **kwargs,
    ):
        """
        Executes a specified task through the agents in the dynamically constructed flow.

        Args:
            task (str): The task for the agents to execute.
            img (Optional[str], optional): An optional image input for the agents.
            imgs (Optional[List[str]], optional): Optional list of images for the agents.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The final result after processing through all agents.

        Raises:
            ValueError: If the task is None or empty.
            Exception: If any error occurs during task execution.
        """
        try:
            # prompt = f"{MULTI_AGENT_COLLAB_PROMPT}\n\n{task}"
            result = self.agent_rearrange.run(
                task=task,
                img=img,
            )

            # Save conversation history after successful execution
            if self.autosave and self.swarm_workspace_dir:
                try:
                    self._save_conversation_history()
                except Exception as e:
                    logger.warning(
                        f"Failed to save conversation history: {e}"
                    )

            return result

        except Exception as e:
            # Save conversation history on error
            if self.autosave and self.swarm_workspace_dir:
                try:
                    self._save_conversation_history()
                except Exception as save_error:
                    logger.warning(
                        f"Failed to save conversation history on error: {save_error}"
                    )

            logger.error(
                f"An error occurred while executing the task: {e}"
            )
            raise e

    def __call__(self, task: str, *args, **kwargs):
        """
        Allows the SequentialWorkflow instance to be called as a function.

        Args:
            task (str): The task for the agents to execute.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The final result after processing through all agents.
        """
        return self.run(task, *args, **kwargs)

    def run_batched(self, tasks: List[str]) -> List[str]:
        """
        Executes a batch of tasks through the agents in the dynamically constructed flow.

        Args:
            tasks (List[str]): A list of tasks for the agents to execute.

        Returns:
            List[str]: A list of final results after processing through all agents.

        Raises:
            ValueError: If tasks is None, empty, or contains non-string elements.
            Exception: If any error occurs during task execution.
        """
        if not tasks or not all(
            isinstance(task, str) for task in tasks
        ):
            raise ValueError(
                "Tasks must be a non-empty list of strings"
            )

        try:
            return [self.agent_rearrange.run(task) for task in tasks]
        except Exception as e:
            logger.error(
                f"An error occurred while executing the batch of tasks: {e}"
            )
            raise

    async def run_async(self, task: str) -> str:
        """
        Executes the specified task through the agents in the dynamically constructed flow asynchronously.

        Args:
            task (str): The task for the agents to execute.

        Returns:
            str: The final result after processing through all agents.

        Raises:
            ValueError: If task is None or not a string.
            Exception: If any error occurs during task execution.
        """
        if not task or not isinstance(task, str):
            raise ValueError("Task must be a non-empty string")

        try:
            return await self.agent_rearrange.run_async(task)
        except Exception as e:
            logger.error(
                f"An error occurred while executing the task asynchronously: {e}"
            )
            raise

    async def run_concurrent(self, tasks: List[str]) -> List[str]:
        """
        Executes a batch of tasks through the agents in the dynamically constructed flow concurrently.

        Args:
            tasks (List[str]): A list of tasks for the agents to execute.

        Returns:
            List[str]: A list of final results after processing through all agents.

        Raises:
            ValueError: If tasks is None, empty, or contains non-string elements.
            Exception: If any error occurs during task execution.
        """
        if not tasks or not all(
            isinstance(task, str) for task in tasks
        ):
            raise ValueError(
                "Tasks must be a non-empty list of strings"
            )

        try:
            with ThreadPoolExecutor(
                max_workers=os.cpu_count()
            ) as executor:
                results = [
                    executor.submit(self.agent_rearrange.run, task)
                    for task in tasks
                ]
                return [
                    result.result()
                    for result in as_completed(results)
                ]
        except Exception as e:
            logger.error(
                f"An error occurred while executing the batch of tasks concurrently: {e}"
            )
            raise

    def _setup_autosave(self):
        """
        Setup workspace directory for saving conversation history.

        Creates the workspace directory structure if autosave is enabled.
        Only conversation history will be saved to this directory.
        """
        try:
            # Set default workspace directory if not set
            if not os.getenv("WORKSPACE_DIR"):
                default_workspace = os.path.join(
                    os.getcwd(), "agent_workspace"
                )
                os.environ["WORKSPACE_DIR"] = default_workspace
                # Clear the cache so get_workspace_dir() picks up the new value
                get_workspace_dir.cache_clear()
                if self.verbose:
                    loguru_logger.info(
                        f"WORKSPACE_DIR not set, using default: {default_workspace}"
                    )

            class_name = self.__class__.__name__
            swarm_name = self.name or "sequential-workflow"
            self.swarm_workspace_dir = get_swarm_workspace_dir(
                class_name, swarm_name, use_timestamp=True
            )

            if self.swarm_workspace_dir:
                if self.verbose:
                    loguru_logger.info(
                        f"Autosave enabled. Conversation history will be saved to: {self.swarm_workspace_dir}"
                    )
        except Exception as e:
            loguru_logger.warning(
                f"Failed to setup autosave for SequentialWorkflow: {e}"
            )
            # Don't raise - autosave failures shouldn't break initialization
            self.swarm_workspace_dir = None

    def _save_conversation_history(self):
        """
        Save conversation history as a separate JSON file to the workspace directory.

        Saves the conversation history to:
        workspace_dir/swarms/SequentialWorkflow/{workflow-name}-{id}/conversation_history.json
        """
        if not self.swarm_workspace_dir:
            return

        try:
            # Get conversation history from agent_rearrange
            conversation_data = []
            if (
                hasattr(self, "agent_rearrange")
                and self.agent_rearrange
            ):
                if (
                    hasattr(self.agent_rearrange, "conversation")
                    and self.agent_rearrange.conversation
                ):
                    if hasattr(
                        self.agent_rearrange.conversation,
                        "conversation_history",
                    ):
                        conversation_data = (
                            self.agent_rearrange.conversation.conversation_history
                        )
                    elif hasattr(
                        self.agent_rearrange.conversation, "to_dict"
                    ):
                        conversation_data = (
                            self.agent_rearrange.conversation.to_dict()
                        )
                    else:
                        conversation_data = []

            # Create conversation history file path
            conversation_path = os.path.join(
                self.swarm_workspace_dir, "conversation_history.json"
            )

            # Save conversation history as JSON
            with open(conversation_path, "w", encoding="utf-8") as f:
                json.dump(
                    conversation_data,
                    f,
                    indent=2,
                    default=str,
                )

            if self.verbose:
                loguru_logger.debug(
                    f"Saved conversation history to {conversation_path}"
                )
        except Exception as e:
            loguru_logger.warning(
                f"Failed to save conversation history: {e}"
            )
