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

from swarms.agents.heavy_swarm_agents import SwarmVariant
from swarms.prompts.multi_agent_collab_prompt import (
    MULTI_AGENT_COLLAB_PROMPT_TWO,
)
from swarms.structs.agent import Agent
from swarms.structs.agent_rearrange import AgentRearrange
from swarms.structs.concurrent_workflow import ConcurrentWorkflow
from swarms.structs.council_as_judge import CouncilAsAJudge
from swarms.structs.debate_with_judge import DebateWithJudge
from swarms.structs.execution_utils import run_concurrently
from swarms.structs.groupchat import GroupChat
from swarms.structs.heavy_swarm import HeavySwarm
from swarms.structs.hiearchical_swarm import HierarchicalSwarm
from swarms.structs.llm_council import LLMCouncil
from swarms.structs.ma_utils import list_all_agents
from swarms.utils.loguru_logger import initialize_logger
from swarms.structs.majority_voting import MajorityVoting
from swarms.structs.mixture_of_agents import MixtureOfAgents
from swarms.structs.multi_agent_router import MultiAgentRouter
from swarms.structs.planner_worker_swarm import PlannerWorkerSwarm
from swarms.structs.round_robin import RoundRobinSwarm
from swarms.structs.sequential_workflow import SequentialWorkflow
from swarms.structs.serialization import SerializableMixin
from swarms.telemetry.otel import (
    capture_init,
    trace_run,
)
from swarms.utils.generate_id import generate_id
from swarms.utils.litellm_wrapper import empty_usage
from swarms.utils.output_types import OutputType
from swarms.utils.workspace_manager import WorkspaceManager

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


logger = initialize_logger(log_folder="swarm_router")


def _msg_batch_run_error(err: Exception, tb: str) -> str:
    return f"SwarmRouter: Error executing batch task on swarm: {err} Traceback: {tb}"


def _msg_fallback_swarms_not_list(actual_type: str) -> str:
    return (
        f"SwarmRouter fallback_swarms must be a list of swarm type "
        f"strings, got {actual_type}."
    )


def _msg_invalid_fallback_swarm_type(swarm_type, valid_types) -> str:
    return (
        f"SwarmRouter fallback_swarms entry {swarm_type!r} is not a "
        f"valid swarm type. Valid types: {', '.join(valid_types)}"
    )


def _msg_fallback_rearrange_flow_required() -> str:
    return (
        "SwarmRouter fallback_swarms includes 'AgentRearrange', which "
        "requires rearrange_flow to be set."
    )


def _msg_swarm_failed_trying_next(
    swarm_type: str, next_type: str, err: Exception
) -> str:
    return (
        f"SwarmRouter swarm '{swarm_type}' failed: {err}. "
        f"Falling back to '{next_type}'."
    )


def _msg_all_swarms_failed(attempts) -> str:
    tried = "; ".join(
        f"{a['swarm_type']}: {type(a['error']).__name__}: {a['error']}"
        for a in attempts
    )
    return (
        f"SwarmRouter exhausted every swarm type without a successful "
        f"run. Attempts: {tried}"
    )


def _msg_collab_prompt_ignored(swarm_type: str) -> str:
    return (
        f"multi_agent_collab_prompt is ignored for swarm_type={swarm_type!r}; "
        f"it is delivered only by {', '.join(sorted(_COLLAB_PROMPT_SWARM_TYPES))}."
    )


# The only constructors that carry the collaboration preamble to the agents
_COLLAB_PROMPT_SWARM_TYPES = frozenset(
    {"SequentialWorkflow", "AgentRearrange"}
)


SwarmType = Literal[
    "AgentRearrange",
    "MixtureOfAgents",
    "SequentialWorkflow",
    "ConcurrentWorkflow",
    "GroupChat",
    "MultiAgentRouter",
    "HierarchicalSwarm",
    "MajorityVoting",
    "CouncilAsAJudge",
    "HeavySwarm",
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
    swarm_type: SwarmType = Field(
        description="Type of swarm to use",
    )
    rearrange_flow: Optional[str] = Field(
        description="Flow configuration string"
    )
    multi_agent_collab_prompt: bool = Field(
        description="Whether to enable multi-agent collaboration prompts",
    )
    fallback_swarms: Optional[List[SwarmType]] = Field(
        default=None,
        description="Swarm types to try, in order, if the primary swarm fails",
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
    """Single entry point for constructing and running swarm orchestrators.

    ``SwarmRouter`` lets callers configure one router and choose the underlying
    orchestration strategy with ``swarm_type``. It validates the router
    configuration during construction, lazily creates the selected swarm on the
    first execution, and forwards ``run()`` calls to that swarm. The created
    swarm is cached by swarm type, agent identities, and construction-time
    configuration, so repeated executions reuse the same orchestrator instance.

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
            Defaults to ``"SequentialWorkflow"``. The factory currently supports
            ``SequentialWorkflow``, ``ConcurrentWorkflow``, ``AgentRearrange``,
            ``MixtureOfAgents``, ``HierarchicalSwarm``, ``GroupChat``,
            ``MultiAgentRouter``, ``MajorityVoting``, ``CouncilAsAJudge``,
            ``HeavySwarm``, ``LLMCouncil``,
            ``DebateWithJudge``, ``RoundRobin``, and ``PlannerWorkerSwarm``.
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
        fallback_swarms (List[SwarmType], optional): Swarm types to try, in
            order, when the primary ``swarm_type`` raises during a run. Each
            fallback is built from the same agents and configuration. The
            first swarm to complete wins; if every one fails,
            :class:`SwarmRouterRunError` is raised carrying every attempt.
            Defaults to ``None`` (no fallback: the primary swarm's error
            propagates unchanged).
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
        council_judge_model_name (str, optional): Judge model for
            ``CouncilAsAJudge``.
        verbose (bool, optional): Emit info/debug logs (reliability check,
            cache hits, swarm creation). Defaults to ``False``.
        worker_tools (List[Callable], optional): Tools passed to ``HeavySwarm``
            workers.
        chairman_model (str, optional): Chairman model for ``LLMCouncil``.
        director_model_name (str, optional): Model name for the director agent
            when ``swarm_type="HierarchicalSwarm"``. Defaults to ``"gpt-5.4"``.
        director_settings (Optional[Dict[str, Any]], optional): Additional
            ``Agent`` keyword arguments forwarded to the ``HierarchicalSwarm``
            director (e.g. ``system_prompt``, ``temperature``, ``top_p``).
            Overrides ``director_model_name`` and other legacy director
            configuration when the corresponding key is present.

    Attributes:
        agents (List[Union[Agent, Callable]]): The configured agent roster.
        swarm: The lazily-constructed underlying swarm instance (populated on
            first ``run()``).
        active_swarm_type (str): The swarm type that served the most recent
            run — the primary ``swarm_type`` unless a fallback was used.
        usage (dict): Provider token usage summed over every agent the
            router's swarms have run. See :attr:`usage`.
        fallback_attempts (List[dict]): One ``{"swarm_type", "error"}`` entry
            per swarm that failed during the most recent run.
        swarm_workspace_dir (str | None): Autosave workspace, set when
            ``autosave=True``.
        logs (list): Per-instance log buffer.

    Factory-backed swarm types:
        ``SequentialWorkflow``, ``ConcurrentWorkflow``, ``AgentRearrange``,
        ``MixtureOfAgents``, ``HierarchicalSwarm``, ``GroupChat``,
        ``MultiAgentRouter``, ``MajorityVoting``, ``CouncilAsAJudge``,
        ``HeavySwarm``, ``LLMCouncil``,
        ``DebateWithJudge``, ``RoundRobin``, ``PlannerWorkerSwarm``.

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
        id: Optional[str] = None,
        name: str = "swarm-router",
        description: str = "Routes your task to the desired swarm",
        max_loops: int = 1,
        agents: List[Union[Agent, Callable]] = [],
        swarm_type: SwarmType = "SequentialWorkflow",
        autosave: bool = False,
        rearrange_flow: str = None,
        fallback_swarms: Optional[List[SwarmType]] = None,
        output_type: OutputType = "dict",
        multi_agent_collab_prompt: bool = False,
        list_all_agents: bool = False,
        conversation: Any = None,
        agents_config: Optional[Dict[Any, Any]] = None,
        heavy_swarm_question_agent_model_name: str = "gpt-5.4",
        heavy_swarm_worker_model_name: str = "gpt-5.4",
        heavy_swarm_swarm_show_output: bool = True,
        heavy_swarm_variant: SwarmVariant = "default",
        heavy_swarm_max_loops: int = 1,
        heavy_swarm_timeout: int = 900,
        council_judge_model_name: str = "gpt-5.4",  # Add missing model_name attribute
        verbose: bool = False,
        worker_tools: List[Callable] = None,
        chairman_model: str = "gpt-5.1",
        autosave_use_timestamp: bool = True,
        director_model_name: str = "gpt-5.4",
        director_settings: Optional[Dict[str, Any]] = None,
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
           wire collaboration prompts and agent listing.

        Raises:
            SwarmRouterConfigError: If the configuration fails validation.
        """
        self.id = id or generate_id("swarm-router")
        self.name = name
        self.description = description
        self.max_loops = max_loops
        self.agents = agents
        self.swarm_type = swarm_type
        self.autosave = autosave
        self.rearrange_flow = rearrange_flow
        self.fallback_swarms = fallback_swarms
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
        self.autosave_use_timestamp = autosave_use_timestamp
        self.director_model_name = director_model_name
        self.director_settings = director_settings
        self.swarm_workspace_dir = None

        # Initialize swarm factory for O(1) lookup performance
        self._swarm_factory = self._initialize_swarm_factory()
        self._swarm_cache = {}  # Cache for created swarms
        # Built lazily on the first run.
        self.swarm = None
        self.active_swarm_type = swarm_type
        self.fallback_attempts = []

        # Always built: a disabled manager no-ops, an absent one raises.
        self._setup_autosave()

        # Reliability check
        self.reliability_check()

        # Capture the full __init__ configuration if telemetry is enabled.
        capture_init(self)

    def _setup_autosave(self):
        """Create the autosave workspace and write the initial config."""
        self.workspace = WorkspaceManager(
            self,
            name=self.name or "swarm-router",
            use_timestamp=self.autosave_use_timestamp,
            enabled=self.autosave,
        )
        self.swarm_workspace_dir = self.workspace.dir

        if self.swarm_workspace_dir:
            self.workspace.save_config()
            self._log(
                "info",
                _msg_autosave_enabled(self.swarm_workspace_dir),
            )

    def reliability_check(self):
        """Validate the router configuration and finish setup.

        Checks performed (in order):
            * ``swarm_type`` is not ``None``.
            * ``swarm_type`` is a string and one of the valid :data:`SwarmType`
              members.
            * ``rearrange_flow`` is provided when ``swarm_type="AgentRearrange"``.
            * ``max_loops != 0``.
            * ``fallback_swarms``, if given, is a list of valid swarm types,
              and ``rearrange_flow`` is set if it includes ``AgentRearrange``.

        On success, calls :meth:`setup` to apply collaboration prompts and
        agent listing.

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

            if self.fallback_swarms is not None:
                if not isinstance(self.fallback_swarms, list):
                    raise SwarmRouterConfigError(
                        _msg_fallback_swarms_not_list(
                            type(self.fallback_swarms).__name__
                        )
                    )
                for fallback in self.fallback_swarms:
                    if fallback not in valid_swarm_types:
                        raise SwarmRouterConfigError(
                            _msg_invalid_fallback_swarm_type(
                                fallback, valid_swarm_types
                            )
                        )
                if (
                    "AgentRearrange" in self.fallback_swarms
                    and self.rearrange_flow is None
                ):
                    raise SwarmRouterConfigError(
                        _msg_fallback_rearrange_flow_required()
                    )

            self.setup()

        except SwarmRouterConfigError as e:
            self._log(
                "error",
                _msg_config_error(e, traceback.format_exc()),
            )
            raise e

    def setup(self):
        """Apply post-validation configuration to the agent roster.

        Warns when the collaboration preamble cannot be delivered for this
        swarm type. The preamble itself is passed to the swarm at
        construction, and the agent roster is seeded once the swarm exists
        (see :meth:`list_agents_to_eachother`). Called from
        :meth:`reliability_check`; not intended to be called directly.
        """
        if (
            self.multi_agent_collab_prompt is True
            and self.swarm_type not in _COLLAB_PROMPT_SWARM_TYPES
        ):
            # Not self._log, that is gated on verbose and this must always show
            logger.warning(
                _msg_collab_prompt_ignored(self.swarm_type)
            )

    @property
    def usage(self) -> dict:
        """Provider token usage summed over every agent this router has run.

        Adds up :attr:`Agent.usage` for the configured ``agents`` plus any
        agent a built swarm holds on its own — a ``HierarchicalSwarm``
        director, a ``MixtureOfAgents`` aggregator, a judge. Keys:
        ``input_tokens``, ``output_tokens``, ``cached_tokens``,
        ``total_tokens``. Agent totals are lifetime totals, so an agent
        shared with another router contributes what it spent there too.
        Streaming calls are not counted.
        """
        total = empty_usage()
        for agent in self._usage_agents():
            for key, value in agent.usage.items():
                total[key] += value
        return total

    def _usage_agents(self) -> List[Agent]:
        """Every distinct Agent the router or its built swarms hold."""
        candidates = list(self.agents or [])
        for swarm in self._swarm_cache.values():
            for value in vars(swarm).values():
                if isinstance(value, (list, tuple)):
                    candidates.extend(value)
                else:
                    candidates.append(value)

        seen = {}
        for candidate in candidates:
            if isinstance(candidate, Agent):
                seen.setdefault(id(candidate), candidate)
        return list(seen.values())

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

    def _initialize_swarm_factory(self) -> Dict[str, Callable]:
        """Build the dispatch table used to instantiate swarm types.

        Returns:
            Dict[str, Callable]: Mapping from factory-backed ``swarm_type``
                strings to bound factory methods.
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
            "LLMCouncil": self._create_llm_council,
            "DebateWithJudge": self._create_debate_with_judge,
            "RoundRobin": self._create_round_robin_swarm,
            "PlannerWorkerSwarm": self._create_planner_worker_swarm,
        }

    def _base_kwargs(self, **extra) -> Dict[str, Any]:
        """Constructor kwargs shared by most swarm factories.

        Returns the ``name``/``description``/``agents``/``max_loops``/
        ``output_type`` set common to nearly every swarm type; pass ``extra``
        to add or override per-swarm keys (e.g. ``verbose``, ``flow``).
        """
        return {
            "name": self.name,
            "description": self.description,
            "agents": self.agents,
            "max_loops": self.max_loops,
            "output_type": self.output_type,
            **extra,
        }

    def _create_heavy_swarm(self, *args, **kwargs):
        """Create a ``HeavySwarm`` using the router's heavy-swarm settings."""
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
        """Create an ``LLMCouncil`` from the configured agents."""
        return LLMCouncil(
            name=self.name,
            description=self.description,
            council_members=self.agents if self.agents else None,
            output_type=self.output_type,
            verbose=self.verbose,
            chairman_model=self.chairman_model,
        )

    def _create_debate_with_judge(self, *args, **kwargs):
        """Create a ``DebateWithJudge`` from pro, con, and judge agents."""
        return DebateWithJudge(
            pro_agent=self.agents[0],
            con_agent=self.agents[1],
            judge_agent=self.agents[2],
            max_loops=self.max_loops,
            output_type=self.output_type,
            verbose=self.verbose,
        )

    def _create_agent_rearrange(self, *args, **kwargs):
        """Create an ``AgentRearrange`` using ``rearrange_flow``."""
        return AgentRearrange(
            *args,
            **self._base_kwargs(
                flow=self.rearrange_flow,
                collab_prompt=self._collab_preamble(),
            ),
            **kwargs,
        )

    def _create_council_as_judge(self, *args, **kwargs):
        """Create a ``CouncilAsAJudge`` with the configured judge model."""
        return CouncilAsAJudge(
            name=self.name,
            description=self.description,
            model_name=self.council_judge_model_name,
            output_type=self.output_type,
        )

    def _create_hierarchical_swarm(self, *args, **kwargs):
        """Create a ``HierarchicalSwarm`` from the configured agents."""
        return HierarchicalSwarm(
            *args,
            **self._base_kwargs(
                director_model_name=self.director_model_name,
                director_settings=self.director_settings,
            ),
            **kwargs,
        )

    def _create_mixture_of_agents(self, *args, **kwargs):
        """Create a ``MixtureOfAgents`` using the last agent as aggregator."""
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
        """Create a ``MajorityVoting`` swarm from the configured agents."""
        return MajorityVoting(*args, **self._base_kwargs(), **kwargs)

    def _create_group_chat(self, *args, **kwargs):
        """Create a ``GroupChat`` from the configured agents."""
        return GroupChat(
            *args,
            **self._base_kwargs(verbose=self.verbose),
            **kwargs,
        )

    def _create_multi_agent_router(self, *args, **kwargs):
        """Create a ``MultiAgentRouter`` from the configured agents."""
        return MultiAgentRouter(
            name=self.name,
            description=self.description,
            agents=self.agents,
            output_type=self.output_type,
        )

    def _collab_preamble(self) -> Optional[str]:
        """Team context delivered to each agent as a system turn.

        Built here rather than written into the shared conversation because
        structures reset that conversation per task, which would discard
        anything seeded before the run.

        Returns:
            The preamble, or ``None`` when neither option is enabled.
        """
        parts = []

        if self.list_all_agents is True:
            parts.append(
                list_all_agents(
                    agents=self.agents,
                    name=self.name,
                    description=self.description,
                    add_collaboration_prompt=False,
                    add_to_conversation=False,
                )
            )

        if self.multi_agent_collab_prompt is True:
            parts.append(MULTI_AGENT_COLLAB_PROMPT_TWO)

        return "\n\n".join(part for part in parts if part) or None

    def _create_sequential_workflow(self, *args, **kwargs):
        """Create a ``SequentialWorkflow`` from the configured agents."""
        return SequentialWorkflow(
            *args,
            **self._base_kwargs(
                multi_agent_collab_prompt=(
                    self._collab_preamble() is not None
                ),
                collab_prompt=self._collab_preamble(),
            ),
            **kwargs,
        )

    def _create_concurrent_workflow(self, *args, **kwargs):
        """Create a ``ConcurrentWorkflow`` from the configured agents."""
        return ConcurrentWorkflow(
            *args, **self._base_kwargs(), **kwargs
        )

    def _create_round_robin_swarm(self, *args, **kwargs):
        """Create a ``RoundRobinSwarm`` from the configured agents."""
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
        """Create a ``PlannerWorkerSwarm`` from the configured agents."""
        return PlannerWorkerSwarm(
            *args,
            **self._base_kwargs(verbose=self.verbose),
            **kwargs,
        )

    def _compute_swarm_cache_key(
        self, swarm_type: Optional[str] = None
    ):
        """Build a stable cache key for the underlying swarm instance.

        Keyed on swarm_type, agent identities, and construction-time config.
        Per-call task/img/tasks args are intentionally excluded — those are
        passed to ``swarm.run()`` and must not invalidate the cached swarm.

        Args:
            swarm_type (str, optional): The swarm type being keyed. Defaults
                to ``self.swarm_type``; fallbacks pass their own.
        """
        swarm_type = swarm_type or self.swarm_type
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
        return (swarm_type, agent_ids, config_hash)

    def _create_swarm(
        self,
        task: str = None,
        swarm_type: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """Create or return the cached swarm for ``swarm_type``.

        The task is accepted for call-site compatibility but is not part of the
        cache key and is not passed to the factory. Runtime inputs such as
        ``task``, ``tasks``, and ``img`` are forwarded later by :meth:`_run`.

        Args:
            task (str, optional): Runtime task associated with the creation
                request. Defaults to ``None``.
            swarm_type (str, optional): Which swarm to build. Defaults to
                ``self.swarm_type``; :meth:`_run` passes each fallback here.
            *args: Positional arguments forwarded to the selected factory.
            **kwargs: Keyword arguments forwarded to the selected factory.

        Returns:
            Any: The cached or newly instantiated swarm object.

        Raises:
            ValueError: If ``self.swarm_type`` has no registered factory.
            RuntimeError: If the selected factory fails.
        """

        swarm_type = swarm_type or self.swarm_type

        cache_key = self._compute_swarm_cache_key(swarm_type)
        cached = self._swarm_cache.get(cache_key)
        if cached is not None:
            self._log("debug", _msg_swarm_cached(swarm_type))
            return cached

        # Use factory pattern for O(1) lookup
        factory_func = self._swarm_factory.get(swarm_type)
        if factory_func is None:
            raise ValueError(
                _msg_invalid_factory_type(
                    swarm_type, list(self._swarm_factory.keys())
                )
            )

        # Create the swarm using the factory function
        try:
            swarm = factory_func(*args, **kwargs)

            # Cache the created swarm for future use
            self._swarm_cache[cache_key] = swarm

            self._log("info", _msg_swarm_created(swarm_type))
            return swarm

        except Exception as e:
            self._log("error", _msg_factory_failed(swarm_type, e))
            raise RuntimeError(
                _msg_factory_failed(swarm_type, e)
            ) from e

    def list_agents_to_eachother(self):
        """Point ``self.conversation`` at the underlying swarm's conversation.

        Called after each run, because the swarm is built lazily and several
        structures replace their conversation object per task.
        """
        if self.swarm is None:
            return

        if self.active_swarm_type == "SequentialWorkflow":
            self.conversation = getattr(
                self.swarm.agent_rearrange, "conversation", None
            )
        else:
            self.conversation = getattr(
                self.swarm, "conversation", None
            )

    def _run(
        self,
        task: Optional[str] = None,
        tasks: Optional[List[str]] = None,
        img: Optional[str] = None,
        *args,
        **kwargs,
    ) -> Any:
        """Run the selected swarm with a single task, task list, or image.

        Args:
            task (str, optional): Single task passed to ``swarm.run`` as
                ``task`` when ``tasks`` is not provided.
            tasks (List[str], optional): Batch-style task list passed to
                ``swarm.run`` as ``tasks``.
            img (str, optional): Image path, URL, or encoded image data passed
                to ``swarm.run`` as ``img``.
            *args: Positional arguments forwarded while creating the swarm.
            **kwargs: Keyword arguments forwarded to ``swarm.run``.

        Returns:
            Any: Result returned by the underlying swarm.

        Raises:
            SwarmRouterRunError: If ``fallback_swarms`` is set and every swarm
                type, primary and fallbacks alike, fails.
            Exception: Without fallbacks, whatever the swarm raised.
        """
        run_kwargs = {}

        if tasks is not None:
            run_kwargs["tasks"] = tasks
        else:
            run_kwargs["task"] = task

        if img is not None:
            run_kwargs["img"] = img

        chain = [self.swarm_type] + list(self.fallback_swarms or [])
        self.fallback_attempts = []

        for position, swarm_type in enumerate(chain):
            try:
                self.swarm = self._create_swarm(
                    task, swarm_type, *args, **kwargs
                )
                self.active_swarm_type = swarm_type
                result = self.swarm.run(**run_kwargs, **kwargs)
                break
            except Exception as e:
                # No fallbacks configured: the caller sees the original error.
                if len(chain) == 1:
                    raise
                self.fallback_attempts.append(
                    {"swarm_type": swarm_type, "error": e}
                )
                if position + 1 < len(chain):
                    logger.warning(
                        _msg_swarm_failed_trying_next(
                            swarm_type, chain[position + 1], e
                        )
                    )
        else:
            raise SwarmRouterRunError(
                _msg_all_swarms_failed(self.fallback_attempts)
            ) from self.fallback_attempts[-1]["error"]

        self.list_agents_to_eachother()

        # Config is written at init; overwriting it here would lose it.
        self.workspace.save_state()
        self.workspace.save_metadata(
            execution_result=result,
            execution_metadata={
                "task": task if task else None,
                "tasks": tasks if tasks else None,
                "status": "completed",
                "swarm_type": self.active_swarm_type,
                "fallback_attempts": [
                    {
                        "swarm_type": a["swarm_type"],
                        "error": str(a["error"]),
                    }
                    for a in self.fallback_attempts
                ],
            },
        )

        return result

    @trace_run(
        "SwarmRouter.run",
        input_params=("task", "tasks", "img"),
    )
    def run(
        self,
        task: Optional[str] = None,
        img: Optional[str] = None,
        tasks: Optional[List[str]] = None,
        *args,
        **kwargs,
    ) -> Any:
        """Execute work on the configured swarm type.

        Creates the underlying swarm if needed, forwards the supplied runtime
        payload to it, and autosaves state/metadata after successful execution
        when autosave is enabled.

        Args:
            task (str, optional): Single task to execute.
            img (str, optional): Image path, URL, or encoded image data to pass
                to the underlying swarm.
            tasks (List[str], optional): Task list to pass to swarm types that
                accept ``tasks``.
            *args: Positional arguments forwarded while creating the swarm.
            **kwargs: Keyword arguments forwarded to the underlying
                ``swarm.run`` call.

        Returns:
            Any: Result returned by the underlying swarm.

        Raises:
            SwarmRouterRunError: Re-raised when the underlying router run error
                is encountered.
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
        """Call :meth:`run` directly from the router instance.

        Args:
            task (str): Single task to execute.
            img (str, optional): Image path, URL, or encoded image data passed
                to :meth:`run`.
            imgs (List[str], optional): Additional image payload forwarded via
                ``kwargs`` to swarm implementations that support it.
            *args: Positional arguments forwarded to :meth:`run`.
            **kwargs: Keyword arguments forwarded to :meth:`run`.

        Returns:
            Any: Result returned by :meth:`run`.
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
        """Execute each task in ``tasks`` with repeated calls to :meth:`run`.

        Args:
            tasks (List[str]): Tasks to execute sequentially.
            img (str, optional): Image payload passed to each run.
            imgs (List[str], optional): Additional image payload forwarded via
                ``kwargs`` to swarm implementations that support it.
            *args: Positional arguments forwarded to each :meth:`run` call.
            **kwargs: Keyword arguments forwarded to each :meth:`run` call.

        Returns:
            List[Any]: Results in the same order as ``tasks``.

        Raises:
            RuntimeError: If any task execution fails.
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
        tasks: List[str],
        imgs: Optional[List[str]] = None,
        *args,
        **kwargs,
    ) -> List[Any]:
        """Execute ``tasks`` through :meth:`run` in parallel.

        The concurrent counterpart to :meth:`batch_run`, which runs the same
        tasks one after another. Results come back in task order, so element
        ``i`` is always the result for ``tasks[i]``.

        Args:
            tasks (List[str]): Tasks to execute in parallel.
            imgs (List[str], optional): One image per task, paired by
                position. Must be the same length as ``tasks``.
            *args: Positional arguments forwarded to each :meth:`run` call.
            **kwargs: Keyword arguments forwarded to each :meth:`run` call.

        Returns:
            List[Any]: One result per task, in task order.

        Raises:
            ValueError: If ``imgs`` is given and is not one per task.
            Exception: Re-raised if any execution fails.
        """
        self._log(
            "info",
            f"SwarmRouter '{self.name}': Executing {len(tasks)} task(s) concurrently",
        )

        try:
            return run_concurrently(
                self.run, tasks, *args, imgs=imgs, **kwargs
            )
        except Exception as e:
            self._log(
                "error",
                f"Error executing tasks concurrently: {e} Traceback: {traceback.format_exc()}",
            )
            raise e
