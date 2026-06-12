import concurrent.futures
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
from swarms.agents.heavy_swarm_agents import SwarmVariant
from swarms.structs.heavy_swarm import HeavySwarm
from swarms.structs.hiearchical_swarm import HierarchicalSwarm
from swarms.structs.llm_council import LLMCouncil
from swarms.structs.ma_utils import list_all_agents
from swarms.structs.majority_voting import MajorityVoting
from swarms.structs.mixture_of_agents import MixtureOfAgents
from swarms.structs.multi_agent_router import MultiAgentRouter
from swarms.structs.planner_worker_swarm import PlannerWorkerSwarm
from swarms.structs.round_robin import RoundRobinSwarm
from swarms.structs.sequential_workflow import SequentialWorkflow
from swarms.structs.serialization import SerializableMixin
from swarms.utils.generate_keys import generate_api_key
from swarms.utils.output_types import OutputType
from swarms.utils.swarm_autosave import (
    autosave_swarm,
    get_swarm_workspace_dir,
)

_DOCS_URL = "https://docs.swarms.world/api/swarm-router"
_REARRANGE_DOCS_URL = "https://docs.swarms.world/api/agent-rearrange"


def _msg_reliability_init(name: str) -> str:
    return (
        f"[SwarmRouter Reliability Check] Initializing SwarmRouter '{name}'. "
        "Validating required parameters for robust operation.\n"
        "For detailed documentation on SwarmRouter configuration, usage, and available swarm types, "
        f"please visit: {_DOCS_URL}"
    )


def _msg_swarm_type_none() -> str:
    return (
        "SwarmRouter: Swarm type cannot be 'none'. "
        f"Check the docs for all the swarm types available. {_DOCS_URL}"
    )


def _msg_swarm_type_not_string(actual_type: str, valid_types) -> str:
    return (
        f"SwarmRouter: swarm_type must be a string, not {actual_type}. "
        f"Valid types are: {', '.join(valid_types)}. "
        "Use swarm_type='SequentialWorkflow' (string), NOT SwarmType.SequentialWorkflow. "
        f"See {_DOCS_URL}"
    )


def _msg_invalid_swarm_type(swarm_type: str, valid_types) -> str:
    return (
        f"SwarmRouter: Invalid swarm_type '{swarm_type}'. "
        f"Valid types are: {', '.join(valid_types)}. "
        f"See {_DOCS_URL}"
    )


def _msg_rearrange_flow_required() -> str:
    return (
        "SwarmRouter: rearrange_flow cannot be 'none' when using AgentRearrange. "
        f"Check the SwarmRouter docs to learn of required parameters. {_REARRANGE_DOCS_URL}"
    )


def _msg_max_loops_zero() -> str:
    return (
        "SwarmRouter: max_loops cannot be 0. "
        f"Check the docs for all the max_loops available. {_DOCS_URL}"
    )


def _msg_config_error(err: Exception, tb: str) -> str:
    return f"SwarmRouterConfigError: {err} Full Traceback: {tb}"


def _msg_fetch_history_error(err: Exception) -> str:
    return f"Error fetching message history as string: {err}"


def _msg_invalid_factory_type(swarm_type: str, valid_types) -> str:
    return (
        f"Invalid swarm type: {swarm_type}. "
        f"Valid types are: {', '.join(valid_types)}"
    )


def _msg_factory_failed(swarm_type: str, err: Exception) -> str:
    return f"Failed to create swarm {swarm_type}: {err}"


def _msg_swarm_created(swarm_type: str) -> str:
    return f"Successfully created swarm: {swarm_type}"


def _msg_swarm_cached(swarm_type: str) -> str:
    return f"Reusing cached swarm: {swarm_type}"


def _msg_autosave_enabled(workspace_dir: str) -> str:
    return f"Autosave enabled. Swarm workspace: {workspace_dir}"


def _msg_autosave_setup_failed(err: Exception) -> str:
    return f"Failed to setup autosave for SwarmRouter: {err}"


def _msg_autosave_after_exec_failed(err: Exception) -> str:
    return f"Failed to autosave after execution: {err}"


def _msg_run_error(name: str, err: Exception, tb: str) -> str:
    return (
        f"\n[SwarmRouter ERROR] '{name}' failed to execute the task on the selected swarm.\n"
        f"Reason: {err}\n"
        f"Traceback:\n{tb}\n\n"
        "Troubleshooting steps:\n"
        "  - Double-check your SwarmRouter configuration (swarm_type, agents, parameters).\n"
        "  - Ensure all individual agents are properly configured and initialized.\n"
        "  - Review the error message and traceback above for clues.\n\n"
        "For detailed documentation on SwarmRouter configuration, usage, and available swarm types, please visit:\n"
        f"  {_DOCS_URL}\n"
    )


def _msg_batch_run_error(err: Exception, tb: str) -> str:
    return f"SwarmRouter: Error executing batch task on swarm: {err} Traceback: {tb}"


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
    "PlannerWorkerSwarm",
]


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
    multi_agent_collab_prompt: bool = Field(
        description="Whether to enable multi-agent collaboration prompts",
    )
    task: str = Field(
        description="The task to be executed by the swarm",
    )

    class Config:
        arbitrary_types_allowed = True


class SwarmRouterRunError(Exception):
    """Raised when the underlying swarm fails during task execution."""

    pass


class SwarmRouterConfigError(Exception):
    """Raised when SwarmRouter is constructed with an invalid configuration."""

    pass


class SwarmRouter(SerializableMixin):
    """Single-entry-point router that dispatches a task to any supported swarm type.

    ``SwarmRouter`` is the highest-level multi-agent abstraction in the framework.
    Pass a list of agents and a ``swarm_type`` and it constructs the matching
    orchestrator (``SequentialWorkflow``, ``ConcurrentWorkflow``,
    ``HierarchicalSwarm``, ``MixtureOfAgents``, ``HeavySwarm`` etc.) and forwards
    ``run()`` calls to it. This lets you swap architectures without rewriting
    orchestration code.

    Construction performs a reliability check on the configuration and, if
    ``autosave=True``, sets up a workspace directory and saves ``config.json``.
    The underlying swarm is constructed lazily on the first ``run()`` call and
    then cached on the instance — repeated ``run()`` calls reuse it (keyed by
    ``swarm_type``, agent identities, and construction-time config).

    Args:
        id (str, optional): Stable identifier for this router instance.
            Auto-generated if omitted.
        name (str, optional): Human-readable name. Used for log lines and
            autosave directory naming. Defaults to ``"swarm-router"``.
        description (str, optional): Free-text description of what this router
            is for.
        max_loops (int, optional): Number of iterations the underlying swarm
            should run. Semantics vary by swarm type (e.g. for
            ``MixtureOfAgents`` this is the number of layers). Defaults to ``1``.
        agents (List[Union[Agent, Callable]], optional): Agents the swarm will
            use. The exact role of each agent depends on ``swarm_type`` — for
            ``DebateWithJudge`` the first two are debaters and the third is the
            judge; for ``MixtureOfAgents`` the last agent is the aggregator.
        swarm_type (SwarmType, optional): Which orchestrator to instantiate.
            See :data:`SwarmType` for the full list. Defaults to
            ``"SequentialWorkflow"``.
        autosave (bool, optional): When ``True``, save ``config.json`` on init
            and ``state.json`` + ``metadata.json`` after each run to
            ``workspace_dir/swarms/SwarmRouter/{name}-{timestamp}/``. Defaults
            to ``False``.
        autosave_use_timestamp (bool, optional): If ``True`` use a timestamp in
            the autosave directory name; otherwise use a UUID. Defaults to
            ``True``.
        rearrange_flow (str, optional): Required when
            ``swarm_type="AgentRearrange"``. Flow-DSL string like
            ``"A -> B, C -> D"``.
        shared_memory_system (Any, optional): A memory backend to inject into
            every agent's ``long_term_memory``.
        output_type (OutputType, optional): How the final swarm output is
            formatted. Defaults to ``"dict-all-except-first"``.
        multi_agent_collab_prompt (bool, optional): Append the multi-agent
            collaboration prompt to every agent's system prompt. Defaults to
            ``True``.
        list_all_agents (bool, optional): When ``True``, every agent is told
            about every other agent at start of run. Defaults to ``False``.
        conversation (Any, optional): Pre-existing conversation object to seed
            the swarm with.
        agents_config (Dict, optional): Optional config overrides per agent.
        heavy_swarm_question_agent_model_name (str, optional): Model for the
            ``HeavySwarm`` question agent.
        heavy_swarm_worker_model_name (str, optional): Model for ``HeavySwarm``
            workers.
        heavy_swarm_swarm_show_output (bool, optional): Print per-agent output
            for ``HeavySwarm``. Defaults to ``True``.
        heavy_swarm_variant (Literal["default", "medium", "heavy"], optional):
            ``HeavySwarm`` architecture. ``"default"`` → 5 agents,
            ``"medium"`` → 4 agents (Captain + Harper / Benjamin / Lucas),
            ``"heavy"`` → 16 agents (Grok captain + 15 specialists). Defaults
            to ``"default"``.
        heavy_swarm_max_loops (int, optional): Iteration count for ``HeavySwarm``
            multi-loop refinement. Defaults to ``1``.
        heavy_swarm_timeout (int, optional): Per-worker wall-clock cap (seconds)
            for ``HeavySwarm``. Defaults to ``900``.
        telemetry_enabled (bool, optional): When ``True`` snapshot each agent's
            config into ``self.agent_config``. Defaults to ``False``.
        council_judge_model_name (str, optional): Judge model for
            ``CouncilAsAJudge``.
        verbose (bool, optional): Emit info/debug logs (reliability check,
            cache hits, swarm creation). Defaults to ``False``.
        worker_tools (List[Callable], optional): Tools passed to ``HeavySwarm``
            workers.
        chairman_model (str, optional): Chairman model for ``LLMCouncil``.

    Attributes:
        agents (List[Union[Agent, Callable]]): The configured agent roster.
        swarm: The lazily-constructed underlying swarm instance (populated on
            first ``run()``).
        swarm_workspace_dir (str | None): Autosave workspace, set when
            ``autosave=True``.
        logs (list): Per-instance log buffer.

    Available swarm types (see :data:`SwarmType` for the canonical list):
        ``SequentialWorkflow``, ``ConcurrentWorkflow``, ``AgentRearrange``,
        ``MixtureOfAgents``, ``HierarchicalSwarm``, ``GroupChat``,
        ``MultiAgentRouter``, ``MajorityVoting``, ``CouncilAsAJudge``,
        ``HeavySwarm``, ``BatchedGridWorkflow``, ``LLMCouncil``,
        ``DebateWithJudge``, ``RoundRobin``, ``PlannerWorkerSwarm``,
        ``AutoSwarmBuilder``, ``auto``.

    Example:
        >>> from swarms import Agent, SwarmRouter
        >>> agents = [
        ...     Agent(agent_name="Researcher", model_name="gpt-5.4"),
        ...     Agent(agent_name="Writer", model_name="gpt-5.4"),
        ... ]
        >>> router = SwarmRouter(agents=agents, swarm_type="SequentialWorkflow")
        >>> result = router.run("Write a brief on transformer architectures.")

    See:
        https://docs.swarms.world/api/swarm-router
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
        shared_memory_system: Any = None,
        output_type: OutputType = "dict-all-except-first",
        multi_agent_collab_prompt: bool = True,
        list_all_agents: bool = False,
        conversation: Any = None,
        agents_config: Optional[Dict[Any, Any]] = None,
        heavy_swarm_question_agent_model_name: str = "gpt-5.4",
        heavy_swarm_worker_model_name: str = "gpt-5.4",
        heavy_swarm_swarm_show_output: bool = True,
        heavy_swarm_variant: SwarmVariant = "default",
        heavy_swarm_max_loops: int = 1,
        heavy_swarm_timeout: int = 900,
        telemetry_enabled: bool = False,
        council_judge_model_name: str = "gpt-5.4",  # Add missing model_name attribute
        verbose: bool = False,
        worker_tools: List[Callable] = None,
        chairman_model: str = "gpt-5.1",
        autosave_use_timestamp: bool = True,
        *args,
        **kwargs,
    ):
        """Initialize the router and validate its configuration.

        See the class docstring for the full parameter list. After assigning
        all fields this method:

        1. Builds the swarm factory dispatch table (``_swarm_factory``) and an
           empty swarm cache (``_swarm_cache``).
        2. If ``autosave=True``, creates the workspace dir and saves
           ``config.json``.
        3. Runs :meth:`reliability_check`, which validates ``swarm_type``,
           ``rearrange_flow``, and ``max_loops``, then calls :meth:`setup` to
           wire shared memory and collaboration prompts.

        Raises:
            SwarmRouterConfigError: If the configuration fails validation.
        """
        self.id = id
        self.name = name
        self.description = description
        self.max_loops = max_loops
        self.agents = agents
        self.swarm_type = swarm_type
        self.autosave = autosave
        self.rearrange_flow = rearrange_flow
        self.shared_memory_system = shared_memory_system
        self.output_type = output_type
        self.logs = []
        self.multi_agent_collab_prompt = multi_agent_collab_prompt
        self.list_all_agents = list_all_agents
        self.conversation = conversation
        self.agents_config = agents_config
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
        self.heavy_swarm_swarm_show_output = (
            heavy_swarm_swarm_show_output
        )
        self.heavy_swarm_variant = heavy_swarm_variant
        self.heavy_swarm_max_loops = heavy_swarm_max_loops
        self.heavy_swarm_timeout = heavy_swarm_timeout
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
                self._log(
                    "info",
                    _msg_autosave_enabled(self.swarm_workspace_dir),
                )
        except Exception as e:
            self._log("warning", _msg_autosave_setup_failed(e))
            # Don't raise - autosave failures shouldn't break initialization
            self.swarm_workspace_dir = None

    def reliability_check(self):
        """Validate the router configuration and finish setup.

        Checks performed (in order):
            * ``swarm_type`` is not ``None``.
            * ``swarm_type`` is a string and one of the valid :data:`SwarmType`
              members.
            * ``rearrange_flow`` is provided when ``swarm_type="AgentRearrange"``.
            * ``max_loops != 0``.

        On success, calls :meth:`setup` to apply shared memory, collaboration
        prompts, and agent listing, and (if ``telemetry_enabled``) snapshots
        :meth:`agent_config`.

        Raises:
            SwarmRouterConfigError: If any check above fails. The error is also
                logged with a full traceback before being re-raised.
        """
        try:

            self._log("info", _msg_reliability_init(self.name))

            # Check swarm type first since it affects other validations
            if self.swarm_type is None:
                raise SwarmRouterConfigError(_msg_swarm_type_none())

            # Validate swarm type is a valid string
            valid_swarm_types = get_args(SwarmType)

            if not isinstance(self.swarm_type, str):
                raise SwarmRouterConfigError(
                    _msg_swarm_type_not_string(
                        type(self.swarm_type).__name__,
                        valid_swarm_types,
                    )
                )

            if self.swarm_type not in valid_swarm_types:
                raise SwarmRouterConfigError(
                    _msg_invalid_swarm_type(
                        self.swarm_type, valid_swarm_types
                    )
                )

            if (
                self.swarm_type == "AgentRearrange"
                and self.rearrange_flow is None
            ):
                raise SwarmRouterConfigError(
                    _msg_rearrange_flow_required()
                )

            # Validate max_loops
            if self.max_loops == 0:
                raise SwarmRouterConfigError(_msg_max_loops_zero())

            self.setup()

            if self.telemetry_enabled:
                self.agent_config = self.agent_config()

        except SwarmRouterConfigError as e:
            self._log(
                "error",
                _msg_config_error(e, traceback.format_exc()),
            )
            raise e

    def setup(self):
        """Apply post-validation configuration to the agent roster.

        Conditionally wires up shared memory, appends the multi-agent
        collaboration preamble, and (if enabled) lists every agent to every
        other agent. Called from :meth:`reliability_check`; not intended to be
        called directly.
        """
        # Handle shared memory
        if self.shared_memory_system is not None:
            self.activate_shared_memory()

        if self.multi_agent_collab_prompt is True:
            self.update_system_prompt_for_agent_in_swarm()

        if self.list_all_agents is True:
            self.list_agents_to_eachother()

    def fetch_message_history_as_string(self):
        """Return the underlying swarm's conversation history as a string.

        Reads from ``self.swarm.conversation`` and excludes the first message
        (typically the system prompt). Requires ``run()`` to have been called
        at least once so that ``self.swarm`` exists.

        Returns:
            str | None: The serialized history, or ``None`` if no history is
            available or the read fails.
        """
        try:
            return (
                self.swarm.conversation.return_all_except_first_string()
            )
        except Exception as e:
            self._log("error", _msg_fetch_history_error(e))
            return None

    def activate_shared_memory(self):
        """Attach ``self.shared_memory_system`` to every agent's
        ``long_term_memory``.

        Idempotent — running it twice with the same memory system is a no-op
        for downstream behavior.
        """
        self._log("info", "Activating shared memory with all agents ")

        for agent in self.agents:
            agent.long_term_memory = self.shared_memory_system

        self._log(
            "info", "All agents now have the same memory system"
        )

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
            "PlannerWorkerSwarm": self._create_planner_worker_swarm,
        }

    def _create_heavy_swarm(self, *args, **kwargs):
        """Factory function for HeavySwarm."""
        return HeavySwarm(
            name=self.name,
            description=self.description,
            output_type=self.output_type,
            question_agent_model_name=self.heavy_swarm_question_agent_model_name,
            worker_model_name=self.heavy_swarm_worker_model_name,
            agent_prints_on=self.heavy_swarm_swarm_show_output,
            worker_tools=self.worker_tools,
            show_dashboard=False,
            variant=self.heavy_swarm_variant,
            max_loops=self.heavy_swarm_max_loops,
            timeout=self.heavy_swarm_timeout,
            verbose=self.verbose,
        )

    def _create_llm_council(self, *args, **kwargs):
        """Factory function for LLMCouncil."""
        return LLMCouncil(
            name=self.name,
            description=self.description,
            council_members=self.agents if self.agents else None,
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
            max_loops=self.max_loops,
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
            output_type=self.output_type,
            verbose=self.verbose,
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
            *args,
            **kwargs,
        )

    def _create_planner_worker_swarm(self, *args, **kwargs):
        """Factory function for PlannerWorkerSwarm."""
        return PlannerWorkerSwarm(
            name=self.name,
            description=self.description,
            agents=self.agents,
            max_loops=self.max_loops,
            output_type=self.output_type,
            verbose=self.verbose,
            *args,
            **kwargs,
        )

    def _compute_swarm_cache_key(self):
        """Build a stable cache key for the underlying swarm instance.

        Keyed on swarm_type, agent identities, and construction-time config.
        Per-call task/img/tasks args are intentionally excluded — those are
        passed to ``swarm.run()`` and must not invalidate the cached swarm.
        """
        agent_ids = tuple(
            getattr(a, "agent_name", None)
            or getattr(a, "__name__", None)
            or id(a)
            for a in (self.agents or [])
        )
        config = (
            self.name,
            self.description,
            self.max_loops,
            self.output_type,
            self.rearrange_flow,
            (
                id(self.shared_memory_system)
                if self.shared_memory_system is not None
                else None
            ),
            self.verbose,
            self.chairman_model,
            self.heavy_swarm_question_agent_model_name,
            self.heavy_swarm_worker_model_name,
            self.heavy_swarm_swarm_show_output,
            self.heavy_swarm_variant,
            self.heavy_swarm_max_loops,
            self.heavy_swarm_timeout,
            self.council_judge_model_name,
        )
        try:
            config_hash = hash(config)
        except TypeError:
            config_hash = hash(repr(config))
        return (self.swarm_type, agent_ids, config_hash)

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

        cache_key = self._compute_swarm_cache_key()
        cached = self._swarm_cache.get(cache_key)
        if cached is not None:
            self._log("debug", _msg_swarm_cached(self.swarm_type))
            return cached

        # Use factory pattern for O(1) lookup
        factory_func = self._swarm_factory.get(self.swarm_type)
        if factory_func is None:
            raise ValueError(
                _msg_invalid_factory_type(
                    self.swarm_type, list(self._swarm_factory.keys())
                )
            )

        # Create the swarm using the factory function
        try:
            swarm = factory_func(*args, **kwargs)

            # Cache the created swarm for future use
            self._swarm_cache[cache_key] = swarm

            self._log("info", _msg_swarm_created(self.swarm_type))
            return swarm

        except Exception as e:
            self._log(
                "error", _msg_factory_failed(self.swarm_type, e)
            )
            raise RuntimeError(
                _msg_factory_failed(self.swarm_type, e)
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
                self._log(
                    "warning",
                    _msg_autosave_after_exec_failed(e),
                )

        return result

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
            self._log(
                "info",
                f"SwarmRouter '{self.name}': Executing task: {task}",
            )
            return self._run(
                task=task,
                img=img,
                tasks=tasks,
                *args,
                **kwargs,
            )
        except SwarmRouterRunError as e:
            self._log(
                "error",
                f"Error executing task: {e} Traceback: {traceback.format_exc()}",
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
        result = self.run(
            task=task, img=img, imgs=imgs, *args, **kwargs
        )
        return result

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
        self._log("info", f"Executing batch of tasks: {tasks}")
        try:
            results = []
            for task in tasks:
                try:
                    result = self.run(
                        task, img=img, imgs=imgs, *args, **kwargs
                    )
                    results.append(result)
                except Exception as e:
                    raise RuntimeError(
                        _msg_batch_run_error(
                            e, traceback.format_exc()
                        )
                    )
            return results
        except Exception as e:
            self._log(
                "error",
                f"Error executing batch of tasks: {e} Traceback: {traceback.format_exc()}",
            )
            raise e

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
        self._log("info", f"Executing task concurrently: {task}")

        try:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=os.cpu_count()
            ) as executor:
                future = executor.submit(
                    self.run,
                    task,
                    img=img,
                    imgs=imgs,
                    *args,
                    **kwargs,
                )
                result = future.result()
                return result
        except Exception as e:
            self._log(
                "error",
                f"Error executing task concurrently: {e} Traceback: {traceback.format_exc()}",
            )
            raise e
