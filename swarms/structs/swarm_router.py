import concurrent.futures
import json
import os
import traceback
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from swarms.prompts.multi_agent_collab_prompt import (
    MULTI_AGENT_COLLAB_PROMPT_TWO,
)
from swarms.structs.agent import Agent
from swarms.structs.concurrent_workflow import ConcurrentWorkflow
from swarms.structs.csv_to_agent import AgentLoader
from swarms.structs.groupchat import GroupChat
from swarms.structs.hiearchical_swarm import HierarchicalSwarm
from swarms.structs.majority_voting import MajorityVoting
from swarms.structs.mixture_of_agents import MixtureOfAgents
from swarms.structs.multi_agent_router import MultiAgentRouter
from swarms.structs.rearrange import AgentRearrange
from swarms.structs.sequential_workflow import SequentialWorkflow
from swarms.structs.spreadsheet_swarm import SpreadSheetSwarm
from swarms.structs.swarm_matcher import swarm_matcher
from swarms.telemetry.log_executions import log_execution
from swarms.utils.output_types import OutputType
from swarms.utils.loguru_logger import initialize_logger
from swarms.structs.malt import MALT
from swarms.structs.deep_research_swarm import DeepResearchSwarm
from swarms.structs.council_judge import CouncilAsAJudge
from swarms.structs.interactive_groupchat import InteractiveGroupChat
from swarms.structs.ma_utils import list_all_agents
from swarms.utils.generate_keys import generate_api_key

logger = initialize_logger(log_folder="swarm_router")

SwarmType = Literal[
    "AgentRearrange",
    "MixtureOfAgents",
    "SpreadSheetSwarm",
    "SequentialWorkflow",
    "ConcurrentWorkflow",
    "GroupChat",
    "MultiAgentRouter",
    "AutoSwarmBuilder",
    "HiearchicalSwarm",
    "auto",
    "MajorityVoting",
    "MALT",
    "DeepResearchSwarm",
    "CouncilAsAJudge",
    "InteractiveGroupChat",
]


class Document(BaseModel):
    file_path: str
    data: str


class SwarmRouterConfig(BaseModel):
    """Configuration model for SwarmRouter."""

    name: str = Field(
        description="Name identifier for the SwarmRouter instance",
    )
    description: str = Field(
        description="Description of the SwarmRouter's purpose",
    )
    # max_loops: int = Field(
    #     description="Maximum number of execution loops"
    # )
    swarm_type: SwarmType = Field(
        description="Type of swarm to use",
    )
    rearrange_flow: Optional[str] = Field(
        description="Flow configuration string"
    )
    rules: Optional[str] = Field(
        description="Rules to inject into every agent"
    )
    multi_agent_collab_prompt: bool = Field(
        description="Whether to enable multi-agent collaboration prompts",
    )
    task: str = Field(
        description="The task to be executed by the swarm",
    )

    class Config:
        arbitrary_types_allowed = True


class SwarmRouter:
    """
    A class that dynamically routes tasks to different swarm types based on user selection or automatic matching.

    The SwarmRouter enables flexible task execution by either using a specified swarm type or automatically determining
    the most suitable swarm type for a given task. It handles task execution while managing logging, type validation,
    and metadata capture.

    Args:
        name (str, optional): Name identifier for the SwarmRouter instance. Defaults to "swarm-router".
        description (str, optional): Description of the SwarmRouter's purpose. Defaults to "Routes your task to the desired swarm".
        max_loops (int, optional): Maximum number of execution loops. Defaults to 1.
        agents (List[Union[Agent, Callable]], optional): List of Agent objects or callables to use. Defaults to empty list.
        swarm_type (SwarmType, optional): Type of swarm to use. Defaults to "SequentialWorkflow".
        autosave (bool, optional): Whether to enable autosaving. Defaults to False.
        flow (str, optional): Flow configuration string. Defaults to None.
        return_json (bool, optional): Whether to return results as JSON. Defaults to False.
        auto_generate_prompts (bool, optional): Whether to auto-generate agent prompts. Defaults to False.
        shared_memory_system (Any, optional): Shared memory system for agents. Defaults to None.
        rules (str, optional): Rules to inject into every agent. Defaults to None.
        documents (List[str], optional): List of document file paths to use. Defaults to empty list.
        output_type (str, optional): Output format type. Defaults to "string". Supported: 'str', 'string', 'list', 'json', 'dict', 'yaml', 'xml'.

    Attributes:
        name (str): Name identifier for the SwarmRouter instance
        description (str): Description of the SwarmRouter's purpose
        max_loops (int): Maximum number of execution loops
        agents (List[Union[Agent, Callable]]): List of Agent objects or callables
        swarm_type (SwarmType): Type of swarm being used
        autosave (bool): Whether autosaving is enabled
        flow (str): Flow configuration string
        return_json (bool): Whether results are returned as JSON
        auto_generate_prompts (bool): Whether prompt auto-generation is enabled
        shared_memory_system (Any): Shared memory system for agents
        rules (str): Rules injected into every agent
        documents (List[str]): List of document file paths
        output_type (str): Output format type. Supported: 'str', 'string', 'list', 'json', 'dict', 'yaml', 'xml'.
        logs (List[SwarmLog]): List of execution logs
        swarm: The instantiated swarm object

    Available Swarm Types:
        - AgentRearrange: Optimizes agent arrangement for task execution
        - MixtureOfAgents: Combines multiple agent types for diverse tasks
        - SpreadSheetSwarm: Uses spreadsheet-like operations for task management
        - SequentialWorkflow: Executes tasks sequentially
        - ConcurrentWorkflow: Executes tasks in parallel
        - "auto": Automatically selects best swarm type via embedding search

    Methods:
        run(task: str, device: str = "cpu", all_cores: bool = False, all_gpus: bool = False, *args, **kwargs) -> Any:
            Executes a task using the configured swarm

        batch_run(tasks: List[str], *args, **kwargs) -> List[Any]:
            Executes multiple tasks in sequence

        threaded_run(task: str, *args, **kwargs) -> Any:
            Executes a task in a separate thread

        async_run(task: str, *args, **kwargs) -> Any:
            Executes a task asynchronously

        concurrent_run(task: str, *args, **kwargs) -> Any:
            Executes a task using concurrent execution

        concurrent_batch_run(tasks: List[str], *args, **kwargs) -> List[Any]:
            Executes multiple tasks concurrently

    """

    def __init__(
        self,
        id: str = generate_api_key(prefix="swarm-router"),
        name: str = "swarm-router",
        description: str = "Routes your task to the desired swarm",
        max_loops: int = 1,
        agents: List[Union[Agent, Callable]] = [],
        swarm_type: SwarmType = "SequentialWorkflow",  # "SpreadSheetSwarm" # "auto"
        autosave: bool = False,
        rearrange_flow: str = None,
        return_json: bool = False,
        auto_generate_prompts: bool = False,
        shared_memory_system: Any = None,
        rules: str = None,
        documents: List[str] = [],  # A list of docs file paths
        output_type: OutputType = "dict-all-except-first",
        speaker_fn: callable = None,
        load_agents_from_csv: bool = False,
        csv_file_path: str = None,
        return_entire_history: bool = True,
        multi_agent_collab_prompt: bool = True,
        list_all_agents: bool = False,
        conversation: Any = None,
        agents_config: Optional[Dict[Any, Any]] = None,
        speaker_function: str = None,
        *args,
        **kwargs,
    ):
        self.id = id
        self.name = name
        self.description = description
        self.max_loops = max_loops
        self.agents = agents
        self.swarm_type = swarm_type
        self.autosave = autosave
        self.rearrange_flow = rearrange_flow
        self.return_json = return_json
        self.auto_generate_prompts = auto_generate_prompts
        self.shared_memory_system = shared_memory_system
        self.rules = rules
        self.documents = documents
        self.output_type = output_type
        self.speaker_fn = speaker_fn
        self.logs = []
        self.load_agents_from_csv = load_agents_from_csv
        self.csv_file_path = csv_file_path
        self.return_entire_history = return_entire_history
        self.multi_agent_collab_prompt = multi_agent_collab_prompt
        self.list_all_agents = list_all_agents
        self.conversation = conversation
        self.agents_config = agents_config
        self.speaker_function = speaker_function

        # Reliability check
        self.reliability_check()

        # Load agents from CSV
        if self.load_agents_from_csv:
            self.agents = AgentLoader(
                csv_path=self.csv_file_path
            ).load_agents()

        self.agent_config = self.agent_config()

    def setup(self):
        if self.auto_generate_prompts is True:
            self.activate_ape()

        # Handle shared memory
        if self.shared_memory_system is not None:
            self.activate_shared_memory()

        # Handle rules
        if self.rules is not None:
            self.handle_rules()

    def activate_shared_memory(self):
        logger.info("Activating shared memory with all agents ")

        for agent in self.agents:
            agent.long_term_memory = self.shared_memory_system

        logger.info("All agents now have the same memory system")

    def handle_rules(self):
        logger.info("Injecting rules to every agent!")

        for agent in self.agents:
            agent.system_prompt += f"### Swarm Rules ### {self.rules}"

        logger.info("Finished injecting rules")

    def activate_ape(self):
        """Activate automatic prompt engineering for agents that support it"""
        try:
            logger.info("Activating automatic prompt engineering...")
            activated_count = 0
            for agent in self.agents:
                if hasattr(agent, "auto_generate_prompt"):
                    agent.auto_generate_prompt = (
                        self.auto_generate_prompts
                    )
                    activated_count += 1
                    logger.debug(
                        f"Activated APE for agent: {agent.name if hasattr(agent, 'name') else 'unnamed'}"
                    )

            logger.info(
                f"Successfully activated APE for {activated_count} agents"
            )

        except Exception as e:
            error_msg = f"Error activating automatic prompt engineering: {str(e)}"
            logger.error(
                f"Error activating automatic prompt engineering in SwarmRouter: {str(e)}"
            )
            raise RuntimeError(error_msg) from e

    def reliability_check(self):
        """Perform reliability checks on swarm configuration.

        Validates essential swarm parameters and configuration before execution.
        Handles special case for CouncilAsAJudge which may not require agents.
        """

        # Check swarm type first since it affects other validations
        if self.swarm_type is None:
            raise ValueError(
                "SwarmRouter: Swarm type cannot be 'none'."
            )

        if self.agents is None:
            raise ValueError(
                "SwarmRouter: No agents provided for the swarm."
            )

        # Validate max_loops
        if self.max_loops == 0:
            raise ValueError("SwarmRouter: max_loops cannot be 0.")

        self.setup()

    def _create_swarm(self, task: str = None, *args, **kwargs):
        """
        Dynamically create and return the specified swarm type or automatically match the best swarm type for a given task.

        Args:
            task (str, optional): The task to be executed by the swarm. Defaults to None.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Union[AgentRearrange, MixtureOfAgents, SpreadSheetSwarm, SequentialWorkflow, ConcurrentWorkflow]:
                The instantiated swarm object.

        Raises:
            ValueError: If an invalid swarm type is provided.
        """
        if self.swarm_type == "auto":
            self.swarm_type = str(swarm_matcher(task))

            self._create_swarm(self.swarm_type)

        elif self.swarm_type == "AgentRearrange":
            return AgentRearrange(
                name=self.name,
                description=self.description,
                agents=self.agents,
                max_loops=self.max_loops,
                flow=self.rearrange_flow,
                return_json=self.return_json,
                output_type=self.output_type,
                return_entire_history=self.return_entire_history,
                *args,
                **kwargs,
            )

        elif self.swarm_type == "MALT":
            return MALT(
                name=self.name,
                description=self.description,
                max_loops=self.max_loops,
                return_dict=True,
                preset_agents=True,
            )

        elif self.swarm_type == "CouncilAsAJudge":
            return CouncilAsAJudge(
                name=self.name,
                description=self.description,
                model_name=self.model_name,
                output_type=self.output_type,
                base_agent=self.agents[0] if self.agents else None,
            )

        elif self.swarm_type == "InteractiveGroupChat":
            return InteractiveGroupChat(
                name=self.name,
                description=self.description,
                agents=self.agents,
                max_loops=self.max_loops,
                output_type=self.output_type,
                speaker_function=self.speaker_function,
            )

        elif self.swarm_type == "DeepResearchSwarm":
            return DeepResearchSwarm(
                name=self.name,
                description=self.description,
                agents=self.agents,
                max_loops=self.max_loops,
                output_type=self.output_type,
            )

        elif self.swarm_type == "HiearchicalSwarm":
            return HierarchicalSwarm(
                name=self.name,
                description=self.description,
                # director=self.agents[0],
                agents=self.agents,
                max_loops=self.max_loops,
                return_all_history=self.return_entire_history,
                output_type=self.output_type,
                *args,
                **kwargs,
            )
        elif self.swarm_type == "MixtureOfAgents":
            return MixtureOfAgents(
                name=self.name,
                description=self.description,
                agents=self.agents,
                aggregator_agent=self.agents[-1],
                layers=self.max_loops,
                output_type=self.output_type,
                *args,
                **kwargs,
            )

        elif self.swarm_type == "MajorityVoting":
            return MajorityVoting(
                name=self.name,
                description=self.description,
                agents=self.agents,
                consensus_agent=self.agents[-1],
                *args,
                **kwargs,
            )
        elif self.swarm_type == "GroupChat":
            return GroupChat(
                name=self.name,
                description=self.description,
                agents=self.agents,
                max_loops=self.max_loops,
                speaker_fn=self.speaker_fn,
                *args,
                **kwargs,
            )

        elif self.swarm_type == "MultiAgentRouter":
            return MultiAgentRouter(
                name=self.name,
                description=self.description,
                agents=self.agents,
                shared_memory_system=self.shared_memory_system,
                output_type=self.output_type,
            )
        elif self.swarm_type == "SpreadSheetSwarm":
            return SpreadSheetSwarm(
                name=self.name,
                description=self.description,
                agents=self.agents,
                max_loops=self.max_loops,
                autosave_on=self.autosave,
                *args,
                **kwargs,
            )
        elif self.swarm_type == "SequentialWorkflow":
            return SequentialWorkflow(
                name=self.name,
                description=self.description,
                agents=self.agents,
                max_loops=self.max_loops,
                shared_memory_system=self.shared_memory_system,
                output_type=self.output_type,
                return_json=self.return_json,
                return_entire_history=self.return_entire_history,
                *args,
                **kwargs,
            )
        elif self.swarm_type == "ConcurrentWorkflow":
            return ConcurrentWorkflow(
                name=self.name,
                description=self.description,
                agents=self.agents,
                max_loops=self.max_loops,
                auto_save=self.autosave,
                return_str_on=self.return_entire_history,
                output_type=self.output_type,
                *args,
                **kwargs,
            )
        else:
            raise ValueError(
                f"Invalid swarm type: {self.swarm_type} try again with a valid swarm type such as 'SequentialWorkflow' or 'ConcurrentWorkflow' or 'auto' or 'AgentRearrange' or 'MixtureOfAgents' or 'SpreadSheetSwarm'"
            )

    def update_system_prompt_for_agent_in_swarm(self):
        # Use list comprehension for faster iteration
        for agent in self.agents:
            if agent.system_prompt is None:
                agent.system_prompt = ""
            agent.system_prompt += MULTI_AGENT_COLLAB_PROMPT_TWO

    def agent_config(self):
        agent_config = {}
        for agent in self.agents:
            agent_config[agent.agent_name] = agent.to_dict()

        return agent_config

    def _run(
        self,
        task: str,
        img: Optional[str] = None,
        model_response: Optional[str] = None,
        imgs: Optional[List[str]] = None,
        *args,
        **kwargs,
    ) -> Any:
        """
        Dynamically run the specified task on the selected or matched swarm type.

        Args:
            task (str): The task to be executed by the swarm.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The result of the swarm's execution.

        Raises:
            Exception: If an error occurs during task execution.
        """
        self.swarm = self._create_swarm(task, *args, **kwargs)

        if self.swarm_type == "SequentialWorkflow":
            self.conversation = (
                self.swarm.agent_rearrange.conversation
            )
        else:
            self.conversation = self.swarm.conversation

        if self.list_all_agents is True:
            list_all_agents(
                agents=self.agents,
                conversation=self.swarm.conversation,
                name=self.name,
                description=self.description,
                add_collaboration_prompt=True,
                add_to_conversation=True,
            )

        if self.multi_agent_collab_prompt is True:
            self.update_system_prompt_for_agent_in_swarm()

        log_execution(
            swarm_id=self.id,
            status="start",
            swarm_config=self.to_dict(),
            swarm_architecture="swarm_router",
        )

        try:
            if self.swarm_type == "CouncilAsAJudge":
                result = self.swarm.run(
                    task=task,
                    img=img,
                    imgs=imgs,
                    model_response=model_response,
                    *args,
                    **kwargs,
                )
            else:
                result = self.swarm.run(task=task, *args, **kwargs)

            log_execution(
                swarm_id=self.id,
                status="completion",
                swarm_config=self.to_dict(),
                swarm_architecture="swarm_router",
            )

            return result
        except Exception as e:
            raise RuntimeError(
                f"SwarmRouter: Error executing task on swarm: {str(e)} Traceback: {traceback.format_exc()}"
            )

    def run(
        self,
        task: str,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
        model_response: Optional[str] = None,
        *args,
        **kwargs,
    ) -> Any:
        """
        Execute a task on the selected swarm type with specified compute resources.

        Args:
            task (str): The task to be executed by the swarm.
            device (str, optional): Device to run on - "cpu" or "gpu". Defaults to "cpu".
            all_cores (bool, optional): Whether to use all CPU cores. Defaults to True.
            all_gpus (bool, optional): Whether to use all available GPUs. Defaults to False.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The result of the swarm's execution.

        Raises:
            Exception: If an error occurs during task execution.
        """
        try:
            return self._run(
                task=task,
                img=img,
                imgs=imgs,
                model_response=model_response,
                *args,
                **kwargs,
            )
        except Exception as e:
            raise RuntimeError(
                f"SwarmRouter: Error executing task on swarm: {str(e)} Traceback: {traceback.format_exc()}"
            )

    def __call__(
        self,
        task: str,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
        *args,
        **kwargs,
    ) -> Any:
        """
        Make the SwarmRouter instance callable.

        Args:
            task (str): The task to be executed by the swarm.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The result of the swarm's execution.
        """
        return self.run(
            task=task, img=img, imgs=imgs, *args, **kwargs
        )

    def batch_run(
        self,
        tasks: List[str],
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
        *args,
        **kwargs,
    ) -> List[Any]:
        """
        Execute a batch of tasks on the selected or matched swarm type.

        Args:
            tasks (List[str]): A list of tasks to be executed by the swarm.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            List[Any]: A list of results from the swarm's execution.

        Raises:
            Exception: If an error occurs during task execution.
        """
        results = []
        for task in tasks:
            try:
                result = self.run(
                    task, img=img, imgs=imgs, *args, **kwargs
                )
                results.append(result)
            except Exception as e:
                raise RuntimeError(
                    f"SwarmRouter: Error executing batch task on swarm: {str(e)} Traceback: {traceback.format_exc()}"
                )
        return results

    def concurrent_run(
        self,
        task: str,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
        *args,
        **kwargs,
    ) -> Any:
        """
        Execute a task on the selected or matched swarm type concurrently.

        Args:
            task (str): The task to be executed by the swarm.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The result of the swarm's execution.

        Raises:
            Exception: If an error occurs during task execution.
        """

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=os.cpu_count()
        ) as executor:
            future = executor.submit(
                self.run, task, img=img, imgs=imgs, *args, **kwargs
            )
            result = future.result()
            return result

    def _serialize_callable(
        self, attr_value: Callable
    ) -> Dict[str, Any]:
        """
        Serializes callable attributes by extracting their name and docstring.

        Args:
            attr_value (Callable): The callable to serialize.

        Returns:
            Dict[str, Any]: Dictionary with name and docstring of the callable.
        """
        return {
            "name": getattr(
                attr_value, "__name__", type(attr_value).__name__
            ),
            "doc": getattr(attr_value, "__doc__", None),
        }

    def _serialize_attr(self, attr_name: str, attr_value: Any) -> Any:
        """
        Serializes an individual attribute, handling non-serializable objects.

        Args:
            attr_name (str): The name of the attribute.
            attr_value (Any): The value of the attribute.

        Returns:
            Any: The serialized value of the attribute.
        """
        try:
            if callable(attr_value):
                return self._serialize_callable(attr_value)
            elif hasattr(attr_value, "to_dict"):
                return (
                    attr_value.to_dict()
                )  # Recursive serialization for nested objects
            else:
                json.dumps(
                    attr_value
                )  # Attempt to serialize to catch non-serializable objects
                return attr_value
        except (TypeError, ValueError):
            return f"<Non-serializable: {type(attr_value).__name__}>"

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts all attributes of the class, including callables, into a dictionary.
        Handles non-serializable attributes by converting them or skipping them.

        Returns:
            Dict[str, Any]: A dictionary representation of the class attributes.
        """
        return {
            attr_name: self._serialize_attr(attr_name, attr_value)
            for attr_name, attr_value in self.__dict__.items()
        }
