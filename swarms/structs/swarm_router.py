import concurrent.futures
import json
import os
import traceback
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Union,
    get_args,
)

from pydantic import BaseModel, Field

from swarms.prompts.multi_agent_collab_prompt import (
    MULTI_AGENT_COLLAB_PROMPT_TWO,
)
from swarms.structs.agent import Agent
from swarms.structs.agent_rearrange import AgentRearrange
from swarms.structs.batched_grid_workflow import BatchedGridWorkflow
from swarms.structs.concurrent_workflow import ConcurrentWorkflow
from swarms.structs.council_as_judge import CouncilAsAJudge
from swarms.structs.debate_with_judge import DebateWithJudge
from swarms.structs.groupchat import GroupChat
from swarms.structs.heavy_swarm import HeavySwarm
from swarms.structs.hiearchical_swarm import HierarchicalSwarm
from swarms.structs.llm_council import LLMCouncil
from swarms.structs.ma_utils import list_all_agents
from swarms.structs.majority_voting import MajorityVoting
from swarms.structs.mixture_of_agents import MixtureOfAgents
from swarms.structs.multi_agent_router import MultiAgentRouter
from swarms.structs.round_robin import RoundRobinSwarm
from swarms.structs.sequential_workflow import SequentialWorkflow
from swarms.utils.generate_keys import generate_api_key
from swarms.utils.loguru_logger import initialize_logger
from swarms.utils.output_types import OutputType
from swarms.utils.swarm_autosave import (
    autosave_swarm,
    get_swarm_workspace_dir,
)

logger = initialize_logger(log_folder="swarm_router")

SwarmType = Literal[
    "AgentRearrange",
    "MixtureOfAgents",
    "SequentialWorkflow",
    "ConcurrentWorkflow",
    "GroupChat",
    "MultiAgentRouter",
    "AutoSwarmBuilder",
    "HierarchicalSwarm",
    "auto",
    "MajorityVoting",
    "CouncilAsAJudge",
    "HeavySwarm",
    "BatchedGridWorkflow",
    "LLMCouncil",
    "DebateWithJudge",
    "RoundRobin",
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


class SwarmRouterRunError(Exception):
    """Exception raised when an error occurs during task execution."""

    pass


class SwarmRouterConfigError(Exception):
    """Exception raised when an error occurs during task execution."""

    pass


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
        autosave (bool, optional): Whether to enable autosaving of swarm configuration, state, and metadata.
            When enabled, saves to workspace_dir/swarms/SwarmRouter/{swarm-name}-{timestamp}/.
            Saves config.json on initialization, and state.json + metadata.json after each run.
            Defaults to False.
        autosave_use_timestamp (bool, optional): If True, use timestamp in directory name; if False, use UUID.
            Defaults to True.
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
        - SequentialWorkflow: Executes tasks sequentially
        - ConcurrentWorkflow: Executes tasks in parallel
        - RoundRobin: Executes tasks in a round-robin fashion, cycling through agents
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
        swarm_type: SwarmType = "SequentialWorkflow",  # "ConcurrentWorkflow" # "auto"
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
        heavy_swarm_loops_per_agent: int = 1,
        heavy_swarm_question_agent_model_name: str = "gpt-4.1",
        heavy_swarm_worker_model_name: str = "gpt-4.1",
        heavy_swarm_swarm_show_output: bool = True,
        telemetry_enabled: bool = False,
        council_judge_model_name: str = "gpt-4o-mini",  # Add missing model_name attribute
        verbose: bool = False,
        worker_tools: List[Callable] = None,
        aggregation_strategy: str = "synthesis",
        chairman_model: str = "gpt-5.1",
        autosave_use_timestamp: bool = True,
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
        self.heavy_swarm_loops_per_agent = heavy_swarm_loops_per_agent
        self.heavy_swarm_question_agent_model_name = (
            heavy_swarm_question_agent_model_name
        )
        self.heavy_swarm_worker_model_name = (
            heavy_swarm_worker_model_name
        )
        self.telemetry_enabled = telemetry_enabled
        self.council_judge_model_name = council_judge_model_name  # Add missing model_name attribute
        self.verbose = verbose
        self.worker_tools = worker_tools
        self.aggregation_strategy = aggregation_strategy
        self.heavy_swarm_swarm_show_output = (
            heavy_swarm_swarm_show_output
        )
        self.chairman_model = chairman_model
        self.autosave = autosave
        self.autosave_use_timestamp = autosave_use_timestamp
        self.swarm_workspace_dir = None

        # Initialize swarm factory for O(1) lookup performance
        self._swarm_factory = self._initialize_swarm_factory()
        self._swarm_cache = {}  # Cache for created swarms

        # Setup autosave workspace if enabled
        if self.autosave:
            self._setup_autosave()

        # Reliability check
        self.reliability_check()

    def _setup_autosave(self):
        """
        Setup autosave workspace directory and save initial configuration.

        Creates the workspace directory structure and saves the initial
        configuration if autosave is enabled.
        """
        try:
            class_name = self.__class__.__name__
            swarm_name = self.name or "swarm-router"
            self.swarm_workspace_dir = get_swarm_workspace_dir(
                class_name, swarm_name, self.autosave_use_timestamp
            )

            if self.swarm_workspace_dir:
                # Save initial configuration
                autosave_swarm(
                    self,
                    self.swarm_workspace_dir,
                    save_config=True,
                    save_state=False,
                    save_metadata=False,
                )
                if self.verbose:
                    logger.info(
                        f"Autosave enabled. Swarm workspace: {self.swarm_workspace_dir}"
                    )
        except Exception as e:
            logger.warning(
                f"Failed to setup autosave for SwarmRouter: {e}"
            )
            # Don't raise - autosave failures shouldn't break initialization
            self.swarm_workspace_dir = None

    def reliability_check(self):
        """Perform reliability checks on swarm configuration.

        Validates essential swarm parameters and configuration before execution.
        Handles special case for CouncilAsAJudge which may not require agents.
        """
        try:

            if self.verbose:
                logger.info(
                    f"[SwarmRouter Reliability Check] Initializing SwarmRouter '{self.name}'. "
                    "Validating required parameters for robust operation.\n"
                    "For detailed documentation on SwarmRouter configuration, usage, and available swarm types, "
                    "please visit: https://docs.swarms.world/en/latest/swarms/structs/swarm_router/"
                )

            # Check swarm type first since it affects other validations
            if self.swarm_type is None:
                raise SwarmRouterConfigError(
                    "SwarmRouter: Swarm type cannot be 'none'. Check the docs for all the swarm types available. https://docs.swarms.world/en/latest/swarms/structs/swarm_router/"
                )

            # Validate swarm type is a valid string
            valid_swarm_types = get_args(SwarmType)

            if not isinstance(self.swarm_type, str):
                raise SwarmRouterConfigError(
                    f"SwarmRouter: swarm_type must be a string, not {type(self.swarm_type).__name__}. "
                    f"Valid types are: {', '.join(valid_swarm_types)}. "
                    "Use swarm_type='SequentialWorkflow' (string), NOT SwarmType.SequentialWorkflow. "
                    "See https://docs.swarms.world/en/latest/swarms/structs/swarm_router/"
                )

            if self.swarm_type not in valid_swarm_types:
                raise SwarmRouterConfigError(
                    f"SwarmRouter: Invalid swarm_type '{self.swarm_type}'. "
                    f"Valid types are: {', '.join(valid_swarm_types)}. "
                    "See https://docs.swarms.world/en/latest/swarms/structs/swarm_router/"
                )

            if (
                self.swarm_type == "AgentRearrange"
                and self.rearrange_flow is None
            ):
                raise SwarmRouterConfigError(
                    "SwarmRouter: rearrange_flow cannot be 'none' when using AgentRearrange. Check the SwarmRouter docs to learn of required parameters. https://docs.swarms.world/en/latest/swarms/structs/agent_rearrange/"
                )

            # Validate max_loops
            if self.max_loops == 0:
                raise SwarmRouterConfigError(
                    "SwarmRouter: max_loops cannot be 0. Check the docs for all the max_loops available. https://docs.swarms.world/en/latest/swarms/structs/swarm_router/"
                )

            self.setup()

            if self.telemetry_enabled:
                self.agent_config = self.agent_config()

        except SwarmRouterConfigError as e:
            logger.error(
                f"SwarmRouterConfigError: {str(e)} Full Traceback: {traceback.format_exc()}"
            )
            raise e

    def setup(self):
        if self.auto_generate_prompts is True:
            self.activate_ape()

        # Handle shared memory
        if self.shared_memory_system is not None:
            self.activate_shared_memory()

        # Handle rules
        if self.rules is not None:
            self.handle_rules()

        if self.multi_agent_collab_prompt is True:
            self.update_system_prompt_for_agent_in_swarm()

        if self.list_all_agents is True:
            self.list_agents_to_eachother()

    def fetch_message_history_as_string(self):
        try:
            return (
                self.swarm.conversation.return_all_except_first_string()
            )
        except Exception as e:
            logger.error(
                f"Error fetching message history as string: {str(e)}"
            )
            return None

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

    def _initialize_swarm_factory(self) -> Dict[str, Callable]:
        """
        Initialize the swarm factory with O(1) lookup performance.

        Returns:
            Dict[str, Callable]: Dictionary mapping swarm types to their factory functions.
        """
        return {
            "HeavySwarm": self._create_heavy_swarm,
            "AgentRearrange": self._create_agent_rearrange,
            "CouncilAsAJudge": self._create_council_as_judge,
            "HierarchicalSwarm": self._create_hierarchical_swarm,
            "MixtureOfAgents": self._create_mixture_of_agents,
            "MajorityVoting": self._create_majority_voting,
            "GroupChat": self._create_group_chat,
            "MultiAgentRouter": self._create_multi_agent_router,
            "SequentialWorkflow": self._create_sequential_workflow,
            "ConcurrentWorkflow": self._create_concurrent_workflow,
            "BatchedGridWorkflow": self._create_batched_grid_workflow,
            "LLMCouncil": self._create_llm_council,
            "DebateWithJudge": self._create_debate_with_judge,
            "RoundRobin": self._create_round_robin_swarm,
        }

    def _create_heavy_swarm(self, *args, **kwargs):
        """Factory function for HeavySwarm."""
        return HeavySwarm(
            name=self.name,
            description=self.description,
            output_type=self.output_type,
            loops_per_agent=self.heavy_swarm_loops_per_agent,
            question_agent_model_name=self.heavy_swarm_question_agent_model_name,
            worker_model_name=self.heavy_swarm_worker_model_name,
            agent_prints_on=self.heavy_swarm_swarm_show_output,
            worker_tools=self.worker_tools,
            aggregation_strategy=self.aggregation_strategy,
            show_dashboard=False,
        )

    def _create_llm_council(self, *args, **kwargs):
        """Factory function for LLMCouncil."""
        return LLMCouncil(
            name=self.name,
            description=self.description,
            output_type=self.output_type,
            verbose=self.verbose,
            chairman_model=self.chairman_model,
        )

    def _create_debate_with_judge(self, *args, **kwargs):
        """Factory function for DebateWithJudge."""
        return DebateWithJudge(
            pro_agent=self.agents[0],
            con_agent=self.agents[1],
            judge_agent=self.agents[2],
            max_rounds=self.max_loops,
            output_type=self.output_type,
            verbose=self.verbose,
        )

    def _create_agent_rearrange(self, *args, **kwargs):
        """Factory function for AgentRearrange."""
        return AgentRearrange(
            name=self.name,
            description=self.description,
            agents=self.agents,
            max_loops=self.max_loops,
            flow=self.rearrange_flow,
            output_type=self.output_type,
            *args,
            **kwargs,
        )

    def _create_batched_grid_workflow(self, *args, **kwargs):
        """Factory function for BatchedGridWorkflow."""
        return BatchedGridWorkflow(
            name=self.name,
            description=self.description,
            agents=self.agents,
            max_loops=self.max_loops,
        )

    def _create_council_as_judge(self, *args, **kwargs):
        """Factory function for CouncilAsAJudge."""
        return CouncilAsAJudge(
            name=self.name,
            description=self.description,
            model_name=self.council_judge_model_name,
            output_type=self.output_type,
        )

    def _create_hierarchical_swarm(self, *args, **kwargs):
        """Factory function for HierarchicalSwarm."""
        return HierarchicalSwarm(
            name=self.name,
            description=self.description,
            agents=self.agents,
            max_loops=self.max_loops,
            output_type=self.output_type,
            *args,
            **kwargs,
        )

    def _create_mixture_of_agents(self, *args, **kwargs):
        """Factory function for MixtureOfAgents."""
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

    def _create_majority_voting(self, *args, **kwargs):
        """Factory function for MajorityVoting."""
        return MajorityVoting(
            name=self.name,
            description=self.description,
            agents=self.agents,
            max_loops=self.max_loops,
            output_type=self.output_type,
            *args,
            **kwargs,
        )

    def _create_group_chat(self, *args, **kwargs):
        """Factory function for GroupChat."""
        return GroupChat(
            name=self.name,
            description=self.description,
            agents=self.agents,
            max_loops=self.max_loops,
            speaker_fn=self.speaker_fn,
            *args,
            **kwargs,
        )

    def _create_multi_agent_router(self, *args, **kwargs):
        """Factory function for MultiAgentRouter."""
        return MultiAgentRouter(
            name=self.name,
            description=self.description,
            agents=self.agents,
            shared_memory_system=self.shared_memory_system,
            output_type=self.output_type,
        )

    def _create_sequential_workflow(self, *args, **kwargs):
        """Factory function for SequentialWorkflow."""
        return SequentialWorkflow(
            name=self.name,
            description=self.description,
            agents=self.agents,
            max_loops=self.max_loops,
            shared_memory_system=self.shared_memory_system,
            output_type=self.output_type,
            *args,
            **kwargs,
        )

    def _create_concurrent_workflow(self, *args, **kwargs):
        """Factory function for ConcurrentWorkflow."""
        return ConcurrentWorkflow(
            name=self.name,
            description=self.description,
            agents=self.agents,
            max_loops=self.max_loops,
            output_type=self.output_type,
            *args,
            **kwargs,
        )

    def _create_round_robin_swarm(self, *args, **kwargs):
        """Factory function for RoundRobinSwarm."""
        return RoundRobinSwarm(
            name=self.name,
            description=self.description,
            agents=self.agents,
            max_loops=self.max_loops,
            verbose=self.verbose,
            return_json_on=self.return_json,
            *args,
            **kwargs,
        )

    def _create_swarm(self, task: str = None, *args, **kwargs):
        """
        Dynamically create and return the specified swarm type with O(1) lookup performance.
        Uses factory pattern with caching for optimal performance.

        Args:
            task (str, optional): The task to be executed by the swarm. Defaults to None.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Union[AgentRearrange, MixtureOfAgents, SequentialWorkflow, ConcurrentWorkflow]:
                The instantiated swarm object.

        Raises:
            ValueError: If an invalid swarm type is provided.
        """

        # Check cache first for better performance
        cache_key = (
            f"{self.swarm_type}_{hash(str(args) + str(kwargs))}"
        )
        if cache_key in self._swarm_cache:
            logger.debug(f"Using cached swarm: {self.swarm_type}")
            return self._swarm_cache[cache_key]

        # Use factory pattern for O(1) lookup
        factory_func = self._swarm_factory.get(self.swarm_type)
        if factory_func is None:
            valid_types = list(self._swarm_factory.keys())
            raise ValueError(
                f"Invalid swarm type: {self.swarm_type}. "
                f"Valid types are: {', '.join(valid_types)}"
            )

        # Create the swarm using the factory function
        try:
            swarm = factory_func(*args, **kwargs)

            # Cache the created swarm for future use
            self._swarm_cache[cache_key] = swarm

            logger.info(
                f"Successfully created swarm: {self.swarm_type}"
            )
            return swarm

        except Exception as e:
            logger.error(
                f"Failed to create swarm {self.swarm_type}: {str(e)}"
            )
            raise RuntimeError(
                f"Failed to create swarm {self.swarm_type}: {str(e)}"
            ) from e

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

    def list_agents_to_eachother(self):
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

    def _run(
        self,
        task: Optional[str] = None,
        tasks: Optional[List[str]] = None,
        img: Optional[str] = None,
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

        args = {}

        if tasks is not None:
            args["tasks"] = tasks
        else:
            args["task"] = task

        if img is not None:
            args["img"] = img

        try:
            if self.swarm_type == "BatchedGridWorkflow":
                result = self.swarm.run(**args, **kwargs)
            else:
                result = self.swarm.run(**args, **kwargs)

            # Autosave after successful execution
            if self.autosave and self.swarm_workspace_dir:
                try:
                    autosave_swarm(
                        self,
                        self.swarm_workspace_dir,
                        save_config=False,  # Don't overwrite initial config
                        save_state=True,
                        save_metadata=True,
                        execution_result=result,
                        additional_data={
                            "execution_metadata": {
                                "task": task if task else None,
                                "tasks": tasks if tasks else None,
                                "status": "completed",
                            }
                        },
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to autosave after execution: {e}"
                    )

            return result
        except SwarmRouterRunError as e:
            logger.error(
                f"\n[SwarmRouter ERROR] '{self.name}' failed to execute the task on the selected swarm.\n"
                f"Reason: {str(e)}\n"
                f"Traceback:\n{traceback.format_exc()}\n\n"
                "Troubleshooting steps:\n"
                "  - Double-check your SwarmRouter configuration (swarm_type, agents, parameters).\n"
                "  - Ensure all individual agents are properly configured and initialized.\n"
                "  - Review the error message and traceback above for clues.\n\n"
                "For detailed documentation on SwarmRouter configuration, usage, and available swarm types, please visit:\n"
                "  https://docs.swarms.world/en/latest/swarms/structs/swarm_router/\n"
            )
            raise e

    def run(
        self,
        task: Optional[str] = None,
        img: Optional[str] = None,
        tasks: Optional[List[str]] = None,
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
                tasks=tasks,
                *args,
                **kwargs,
            )
        except SwarmRouterRunError as e:
            logger.error(
                f"\n[SwarmRouter ERROR] '{self.name}' failed to execute the task on the selected swarm.\n"
                f"Reason: {str(e)}\n"
                f"Traceback:\n{traceback.format_exc()}\n\n"
                "Troubleshooting steps:\n"
                "  - Double-check your SwarmRouter configuration (swarm_type, agents, parameters).\n"
                "  - Ensure all individual agents are properly configured and initialized.\n"
                "  - Review the error message and traceback above for clues.\n\n"
                "For detailed documentation on SwarmRouter configuration, usage, and available swarm types, please visit:\n"
                "  https://docs.swarms.world/en/latest/swarms/structs/swarm_router/\n"
            )
            raise e

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
