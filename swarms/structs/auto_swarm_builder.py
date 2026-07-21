import json
import re
import traceback
from typing import Any, List, Optional, Type, get_args

from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel, Field, ValidationError

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.structs.swarm_router import SwarmRouter, SwarmType
from swarms.utils.litellm_wrapper import LiteLLM

load_dotenv()

swarm_types = [
    "return-agents",
    "return-swarm-router-config",
    "return-agents-objects",
]

# swarm_type values that are valid for SwarmRouter in general but make no
# sense as something the boss agent generates for a *sub*-swarm spec:
# "AutoSwarmBuilder" would recurse into itself, and "auto" defers the choice
# rather than making one.
_EXCLUDED_GENERATED_SWARM_TYPES = {"AutoSwarmBuilder", "auto"}

BOSS_SYSTEM_PROMPT = """
You are an expert multi-agent architecture designer and team coordinator. Your role is to create and orchestrate sophisticated teams of specialized AI agents, each with distinct personalities, roles, and capabilities. Your primary goal is to ensure the multi-agent system operates efficiently while maintaining clear communication, well-defined responsibilities, and optimal task distribution.

### Core Design Principles:

1. **Comprehensive Task Analysis**:
   - Thoroughly deconstruct the task into its fundamental components and sub-tasks
   - Identify the specific skills, knowledge domains, and personality traits required for each component
   - Analyze potential challenges, dependencies, and coordination requirements between agents
   - Map out optimal workflows, information flow patterns, and decision-making hierarchies
   - Consider scalability, maintainability, and adaptability requirements

2. **Agent Design Excellence**:
   - Each agent must have a crystal-clear, specific purpose and domain of expertise
   - Design agents with distinct, complementary personalities that enhance team dynamics
   - Ensure agents are self-aware of their limitations and know when to seek assistance
   - Create agents that can effectively communicate progress, challenges, and insights
   - Design for resilience, adaptability, and continuous learning capabilities

3. **Comprehensive Agent Framework**:
   For each agent, meticulously define:
   - **Role & Purpose**: Precise description of responsibilities, authority, and contribution to team objectives
   - **Personality Profile**: Distinct characteristics that influence thinking patterns, communication style, and decision-making approach
   - **Expertise Matrix**: Specific knowledge domains, skill sets, tools, and capabilities
   - **Communication Protocol**: How the agent presents information, interacts with others, and reports progress
   - **Decision-Making Framework**: Systematic approach to problem-solving, risk assessment, and choice evaluation
   - **Limitations & Boundaries**: Clear constraints, ethical guidelines, and operational boundaries
   - **Collaboration Strategy**: How the agent works with others, shares knowledge, and contributes to team success

4. **Advanced System Prompt Engineering**:
   Create comprehensive system prompts that include:
   - Detailed role and purpose explanation with context and scope
   - Rich personality description with behavioral guidelines and interaction patterns
   - Comprehensive capabilities, tools, and resource specifications
   - Detailed communication protocols, reporting requirements, and feedback mechanisms
   - Systematic problem-solving approach with decision-making frameworks
   - Collaboration guidelines, team interaction rules, and conflict resolution procedures
   - Quality standards, success criteria, and performance metrics
   - Error handling, recovery procedures, and escalation protocols

5. **Multi-Agent Coordination Architecture**:
   - Design robust communication channels and protocols between agents
   - Establish clear task handoff procedures and information sharing mechanisms
   - Create feedback loops for continuous improvement and adaptation
   - Implement comprehensive error handling and recovery procedures
   - Define escalation paths for complex issues and decision-making hierarchies
   - Design monitoring, logging, and performance tracking systems

6. **Quality Assurance & Governance**:
   - Set measurable success criteria for each agent and the overall system
   - Implement verification steps, validation procedures, and quality checks
   - Create mechanisms for self-assessment, peer review, and continuous improvement
   - Establish protocols for handling edge cases, unexpected situations, and failures
   - Design governance structures for oversight, accountability, and performance management

### Multi-Agent Architecture Types:

Choose the most appropriate architecture based on task requirements:

- **AgentRearrange**: Dynamic task reallocation based on agent performance and workload
- **MixtureOfAgents**: Parallel processing with specialized agents working independently
- **SpreadSheetSwarm**: Structured data processing with coordinated workflows
- **SequentialWorkflow**: Linear task progression with handoffs between agents
- **ConcurrentWorkflow**: Parallel execution with coordination and synchronization
- **GroupChat**: Collaborative discussion and consensus-building approach
- **MultiAgentRouter**: Intelligent routing and load balancing across agents
- **HierarchicalSwarm**: Layered decision-making with management and execution tiers
- **MajorityVoting**: Democratic decision-making with voting mechanisms
- **CouncilAsAJudge**: Deliberative decision-making with expert panels
- **HeavySwarm**: High-capacity processing with multiple specialized agents

### Output Requirements:

When creating a multi-agent system, provide:

1. **Agent Specifications**:
   - Comprehensive role and purpose statements
   - Detailed personality profiles and behavioral characteristics
   - Complete capabilities, limitations, and boundary definitions
   - Communication style and interaction protocols
   - Collaboration strategies and team integration plans

2. **System Prompts**:
   - Complete, detailed prompts that embody each agent's identity and capabilities
   - Clear behavioral instructions and decision-making frameworks
   - Specific interaction guidelines and reporting requirements
   - Quality standards and performance expectations

3. **Architecture Design**:
   - Team structure, hierarchy, and reporting relationships
   - Communication flow patterns and information routing
   - Task distribution strategies and workload balancing
   - Quality control measures and performance monitoring
   - Error handling and recovery procedures

### Best Practices:

- Prioritize clarity, specificity, and precision in agent design
- Ensure each agent has a unique, well-defined role with clear boundaries
- Create comprehensive, detailed system prompts that leave no ambiguity
- Maintain thorough documentation of agent capabilities, limitations, and interactions
- Design for scalability, adaptability, and long-term maintainability
- Focus on creating agents that work together synergistically and efficiently
- Consider edge cases, failure modes, and contingency planning
- Implement robust error handling, monitoring, and recovery procedures
- Design for continuous learning, improvement, and optimization
- Ensure ethical considerations, safety measures, and responsible AI practices
- Never choose "AutoSwarmBuilder" or "auto" as the swarm_type — always select a concrete architecture from the list above
- Agent names must be unique within the team

### Output Format:

Respond with ONLY valid JSON matching the required schema. Do not wrap the JSON in markdown code fences and do not include any explanatory text before or after it.
"""


class AgentSpec(BaseModel):
    """Configuration for an individual agent specification."""

    agent_name: Optional[str] = Field(
        None,
        description="The unique name assigned to the agent, which identifies its role and functionality within the swarm.",
    )
    description: Optional[str] = Field(
        None,
        description="A detailed explanation of the agent's purpose, capabilities, and any specific tasks it is designed to perform.",
    )
    system_prompt: Optional[str] = Field(
        None,
        description="The initial instruction or context provided to the agent, guiding its behavior and responses during execution.",
    )
    model_name: Optional[str] = Field(
        "gpt-5.4",
        description="The name of the AI model that the agent will utilize for processing tasks and generating outputs. For example: gpt-4o, gpt-4o-mini, openai/o3-mini",
    )
    auto_generate_prompt: Optional[bool] = Field(
        False,
        description="A flag indicating whether the agent should automatically create prompts based on the task requirements.",
    )
    max_tokens: Optional[int] = Field(
        8192,
        description="The maximum number of tokens that the agent is allowed to generate in its responses, limiting output length.",
    )
    temperature: Optional[float] = Field(
        0.5,
        description="A parameter that controls the randomness of the agent's output; lower values result in more deterministic responses.",
    )
    role: Optional[str] = Field(
        "worker",
        description="The designated role of the agent within the swarm, which influences its behavior and interaction with other agents.",
    )
    max_loops: Optional[int] = Field(
        1,
        description="The maximum number of times the agent is allowed to repeat its task, enabling iterative processing if necessary.",
    )
    goal: Optional[str] = Field(
        None,
        description="The primary objective or desired outcome the agent is tasked with achieving.",
    )


class Agents(BaseModel):
    """Configuration for a collection of agents that work together as a swarm to accomplish tasks."""

    agents: List[AgentSpec] = Field(
        description="A list containing the specifications of each agent that will participate in the swarm, detailing their roles and functionalities."
    )


class AgentsConfig(BaseModel):
    """Configuration for a list of agents in a swarm"""

    agents: List[AgentSpec] = Field(
        description="A list of agent configurations",
    )


class SwarmRouterConfig(BaseModel):
    """Configuration model for SwarmRouter.

    This is the single source of truth generated by the boss agent: it
    contains the agent roster, the chosen multi-agent architecture
    (``swarm_type``), and the task, so that whatever consumes it (agent
    construction, ``SwarmRouter`` execution) sees one consistent spec
    instead of results from separate, potentially-disagreeing LLM calls.
    """

    name: str = Field(description="The name of the team of agents")
    description: str = Field(
        description="Description of the team of agents"
    )
    agents: List[AgentSpec] = Field(
        description="A list of agent configurations",
    )
    swarm_type: SwarmType = Field(
        description="Type of multi-agent structure to use",
    )
    rearrange_flow: Optional[str] = Field(
        None,
        description="Flow configuration string. Only to be used if you you use the AgentRearrange multi-agent structure",
    )
    rules: Optional[str] = Field(
        None,
        description="Rules to inject into every agent. This is a string of rules that will be injected into every agent's system prompt. This is a good place to put things like 'You are a helpful assistant' or 'You are a helpful assistant that can answer questions and help with tasks'.",
    )
    multi_agent_collab_prompt: Optional[str] = Field(
        None,
        description="Prompt for multi-agent collaboration and coordination.",
    )
    task: str = Field(
        description="The task to be executed by the swarm",
    )

    class Config:
        arbitrary_types_allowed = True


class AutoSwarmBuilder:
    """Automatically designs, and optionally runs, a multi-agent swarm for a task.

    This class uses a boss LLM to analyze a task and produce a swarm
    specification — the agent roster, the chosen ``swarm_type``, and the
    task — as a single, schema-validated ``SwarmRouterConfig``. That one
    spec is then reused everywhere: to build the real ``Agent`` objects and,
    when requested, to construct and run the ``SwarmRouter`` — so the agents
    that actually execute are guaranteed to match what was reported back to
    the caller, instead of results from two independent, potentially
    disagreeing LLM calls.

    Malformed or schema-invalid LLM output triggers up to
    ``max_json_repair_attempts`` repair retries (the error is fed back to
    the model) before raising.

    Args:
        name (str): The name of the swarm. Defaults to "auto-swarm-builder".
        description (str): A description of the swarm's purpose. Defaults to "Auto Swarm Builder".
        verbose (bool): Whether to output detailed logs. Defaults to True.
        max_loops (int): Maximum number of execution loops for the constructed swarm. Defaults to 1.
        model_name (str): The LLM model to use for the boss agent. Defaults to "gpt-5.4".
        max_tokens (int): Maximum tokens for the boss LLM's responses. Defaults to 8000.
        swarm_type (str): Which spec-only result to return from ``run()`` when
            ``execute`` is not requested. One of "return-agents", "return-swarm-router-config",
            "return-agents-objects". Defaults to "return-agents".
        system_prompt (str): System prompt for the boss agent. Defaults to BOSS_SYSTEM_PROMPT.
        additional_llm_args (dict, optional): Extra kwargs forwarded to the boss LLM.
        max_agents (int, optional): Upper bound on how many agents the boss agent may
            generate; exceeding it raises ``ValueError``. Unbounded when ``None``.
        max_json_repair_attempts (int): Number of repair retries when the boss agent's
            output is malformed JSON or fails schema validation. Defaults to 1.
        auto_execute (bool): Default value for ``run()``'s ``execute`` parameter — when
            True, ``run()`` builds real agents and a ``SwarmRouter`` from the generated
            spec and executes it, instead of returning a spec-only result. Defaults to False.
    """

    def __init__(
        self,
        name: str = "auto-swarm-builder",
        description: str = "Auto Swarm Builder",
        verbose: bool = True,
        max_loops: int = 1,
        model_name: str = "gpt-5.4",
        max_tokens: int = 8000,
        swarm_type: str = "return-agents",
        system_prompt: str = BOSS_SYSTEM_PROMPT,
        additional_llm_args: Optional[dict] = None,
        auto_execute: bool = False,
    ):
        self.name = name
        self.description = description
        self.verbose = verbose
        self.max_loops = max_loops
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.swarm_type = swarm_type
        self.system_prompt = system_prompt
        self.additional_llm_args = additional_llm_args or {}
        self.max_agents = None
        self.max_json_repair_attempts = 1
        self.auto_execute = auto_execute
        self.conversation = Conversation()
        self.agents_pool = []

        self.reliability_check()

    def reliability_check(self):
        """Validate the AutoSwarmBuilder configuration.

        Raises:
            ValueError: If max_loops is 0, max_agents is non-positive when
                set, or max_json_repair_attempts is negative.
        """
        if self.max_loops == 0:
            raise ValueError(
                f"AutoSwarmBuilder: {self.name} max_loops cannot be 0"
            )

        if self.max_agents is not None and self.max_agents <= 0:
            raise ValueError(
                f"AutoSwarmBuilder: {self.name} max_agents must be "
                "greater than 0 when set."
            )

        if self.max_json_repair_attempts < 0:
            raise ValueError(
                f"AutoSwarmBuilder: {self.name} max_json_repair_attempts "
                "must be greater than or equal to 0."
            )

        logger.info(
            f"Initializing AutoSwarmBuilder: {self.name} Description: {self.description}"
        )

    def build_llm_agent(self, config: Type[BaseModel]) -> LiteLLM:
        """Build a LiteLLM agent constrained to a structured-output schema.

        Args:
            config (Type[BaseModel]): Pydantic model the LLM's response must
                conform to (used as ``response_format``).

        Returns:
            LiteLLM: Configured LiteLLM instance.
        """
        return LiteLLM(
            model_name=self.model_name,
            system_prompt=self.system_prompt,
            temperature=0.5,
            response_format=config,
            max_tokens=self.max_tokens,
            **self.additional_llm_args,
        )

    @staticmethod
    def _parse_llm_json(raw: Any) -> dict:
        """Parse an LLM's JSON response, tolerating markdown code fences.

        Args:
            raw: The raw LLM output — normally a JSON string, but passed
                through unchanged if already a dict.

        Returns:
            dict: The parsed JSON payload.

        Raises:
            ValueError: If ``raw`` is neither a dict nor a string.
            json.JSONDecodeError: If the string isn't valid JSON.
        """
        if isinstance(raw, dict):
            return raw
        if not isinstance(raw, str):
            raise ValueError(
                f"Unexpected LLM output type: {type(raw).__name__}"
            )

        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
            text = text.strip()

        return json.loads(text)

    def _generate_structured_output(
        self, config: Type[BaseModel], prompt: str
    ) -> BaseModel:
        """Call the boss LLM for a structured, schema-validated response.

        Retries with an error-specific repair prompt (feeding the parse or
        validation error back to the model) up to
        ``max_json_repair_attempts`` times before raising.

        Args:
            config (Type[BaseModel]): The expected response schema.
            prompt (str): The task/instruction prompt for the boss agent.

        Returns:
            BaseModel: A validated instance of ``config``.

        Raises:
            ValueError: If no valid response is obtained within the retry budget.
        """
        model = self.build_llm_agent(config=config)
        original_prompt = prompt
        current_prompt = prompt
        last_error = None

        for attempt in range(self.max_json_repair_attempts + 1):
            raw = model.run(current_prompt)
            try:
                data = self._parse_llm_json(raw)
                return config.model_validate(data)
            except (
                json.JSONDecodeError,
                ValidationError,
                ValueError,
            ) as e:
                last_error = e
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_json_repair_attempts + 1} "
                    f"for {config.__name__} produced invalid output "
                    f"({type(e).__name__}): {e}"
                )
                current_prompt = (
                    f"Your previous response was invalid: {e}\n\n"
                    "Return ONLY valid JSON that matches the required "
                    f"schema, with no markdown fences or extra text.\n\n"
                    f"{original_prompt}"
                )

        raise ValueError(
            f"Failed to obtain a valid '{config.__name__}' response from "
            f"the boss agent after {self.max_json_repair_attempts + 1} "
            f"attempt(s): {last_error}"
        )

    @staticmethod
    def _check_duplicate_agent_names(
        names: List[Optional[str]],
    ) -> None:
        """Raise if any two non-empty agent names collide.

        Duplicate ``agent_name`` values break ``persistent_memory`` (agents
        share the same ``MEMORY.md`` file, per the framework's own agent
        guidance), so this is enforced wherever a batch of agent specs is
        generated or supplied.

        Args:
            names: Candidate agent names; empty/None entries are ignored.

        Raises:
            ValueError: If a name appears more than once.
        """
        seen = set()
        for name in names:
            if not name:
                continue
            if name in seen:
                raise ValueError(
                    f"Duplicate agent_name '{name}' — agent names must be "
                    "unique within a swarm (duplicate names corrupt "
                    "persistent_memory, which is keyed by agent_name)."
                )
            seen.add(name)

    def _validate_swarm_type(self, swarm_type: str) -> None:
        """Validate a boss-agent-generated ``swarm_type``.

        Args:
            swarm_type: The generated swarm type string.

        Raises:
            ValueError: If it isn't a recognized ``SwarmType``, or is one of
                the types excluded for generated specs (``AutoSwarmBuilder``,
                which would recurse; ``auto``, which defers rather than
                choosing an architecture).
        """
        valid_types = set(get_args(SwarmType))
        if (
            swarm_type not in valid_types
            or swarm_type in _EXCLUDED_GENERATED_SWARM_TYPES
        ):
            allowed = sorted(
                valid_types - _EXCLUDED_GENERATED_SWARM_TYPES
            )
            raise ValueError(
                f"Generated swarm_type '{swarm_type}' is not usable here. "
                f"Valid options: {allowed}"
            )

    def _enforce_max_agents(self, count: int) -> None:
        """Raise if a generated agent count exceeds ``max_agents``."""
        if self.max_agents is not None and count > self.max_agents:
            raise ValueError(
                f"Boss agent generated {count} agents, exceeding "
                f"max_agents={self.max_agents}."
            )

    def create_agents(self, task: str) -> dict:
        """Generate agent specifications for a task.

        Args:
            task (str): The task to create agents for.

        Returns:
            dict: ``{"agents": [...]}`` — validated agent specifications.

        Raises:
            ValueError: If the boss agent's output can't be validated, if
                agent names collide, or if ``max_agents`` is exceeded.
        """
        try:
            logger.info("Creating agents from specifications")
            agents_config: Agents = self._generate_structured_output(
                config=Agents, prompt=task
            )
            self._check_duplicate_agent_names(
                [spec.agent_name for spec in agents_config.agents]
            )
            self._enforce_max_agents(len(agents_config.agents))

            return agents_config.model_dump()

        except Exception as e:
            logger.error(
                f"Error creating agents: {str(e)} Traceback: {traceback.format_exc()}",
                exc_info=True,
            )
            raise

    def _build_validated_router_config(
        self, task: str
    ) -> SwarmRouterConfig:
        """Generate and validate a single ``SwarmRouterConfig`` for a task.

        This is the one place a full swarm spec (agents + swarm_type + task)
        is generated, so every caller that needs a consistent spec —
        :meth:`create_router_config`, :meth:`build_and_run_swarm`,
        :meth:`initialize_swarm_router` — goes through it.

        Args:
            task (str): The task to design a swarm for.

        Returns:
            SwarmRouterConfig: The validated spec, with ``task`` set to the
                exact caller-provided string (not the model's own echo of it).
        """
        config: SwarmRouterConfig = self._generate_structured_output(
            config=SwarmRouterConfig,
            prompt=f"Create the multi-agent team for the following task: {task}",
        )
        self._validate_swarm_type(config.swarm_type)
        self._check_duplicate_agent_names(
            [spec.agent_name for spec in config.agents]
        )
        self._enforce_max_agents(len(config.agents))
        config.task = task
        return config

    def create_router_config(self, task: str) -> dict:
        """Generate a full swarm specification for a task.

        Returns the agent roster, chosen ``swarm_type``, and task as a
        single validated dictionary — from one LLM call.

        Args:
            task (str): The task to design a swarm for.

        Returns:
            dict: The ``SwarmRouterConfig`` fields as a dict, including
                ``agents``, ``swarm_type``, and ``task``.

        Raises:
            ValueError: If the generated spec is invalid.
        """
        try:
            logger.info(
                f"Creating swarm router config for task: {task}"
            )
            config = self._build_validated_router_config(task)
            return config.model_dump()

        except Exception as e:
            logger.error(
                f"Error creating swarm router config: {str(e)} Traceback: {traceback.format_exc()}",
                exc_info=True,
            )
            raise

    def build_and_run_swarm(self, task: str) -> dict:
        """Design a multi-agent team for ``task`` and execute it.

        Generates one validated :class:`SwarmRouterConfig`, builds the real
        ``Agent`` objects from that same spec's agents, and runs a
        ``SwarmRouter`` with them — so the agents that execute are exactly
        the ones described in the returned metadata, unlike chaining
        :meth:`create_agents` and :meth:`initialize_swarm_router` (which
        make two independent, potentially-disagreeing LLM calls).

        Args:
            task (str): The task to design and run a swarm for.

        Returns:
            dict: ``{"name", "description", "agents" (names actually built),
                "swarm_type", "task", "output"}`` where ``output`` is the
                ``SwarmRouter``'s execution result.
        """
        config = self._build_validated_router_config(task)
        agents = self.create_agents_from_specs(
            {"agents": config.agents}
        )

        logger.info(
            f"Executing '{config.swarm_type}' swarm with agents: "
            f"{[a.agent_name for a in agents]}"
        )

        # Note: SwarmRouterConfig.multi_agent_collab_prompt is a free-form
        # str (guidance text for the boss agent), while SwarmRouter's own
        # multi_agent_collab_prompt kwarg is a bool (enable/disable a
        # built-in prompt). The two are not interchangeable, so it is
        # intentionally not forwarded here.
        swarm_router = SwarmRouter(
            name=config.name,
            description=config.description,
            max_loops=self.max_loops,
            swarm_type=config.swarm_type,
            rearrange_flow=config.rearrange_flow,
            rules=config.rules,
            agents=agents,
            output_type="dict",
        )

        output = swarm_router.run(task)

        return {
            "name": config.name,
            "description": config.description,
            "agents": [agent.agent_name for agent in agents],
            "swarm_type": config.swarm_type,
            "task": task,
            "output": output,
        }

    def _execute_task(self, task: str) -> dict:
        """Backward-compatible alias for :meth:`build_and_run_swarm`."""
        return self.build_and_run_swarm(task)

    def dict_to_agent(self, output: Any) -> List[Agent]:
        """Convert a raw ``{"agents": [...]}`` dict into ``Agent`` objects.

        Delegates to :meth:`create_agents_from_specs`, which correctly maps
        ``agent_name``/``description`` -> ``agent_description`` and handles
        both dict and Pydantic ``AgentSpec`` entries.

        Args:
            output (Any): Dictionary containing agent configurations. Any
                non-dict input returns an empty list.

        Returns:
            List[Agent]: The created Agent objects.
        """
        if not isinstance(output, dict):
            return []
        return self.create_agents_from_specs(output)

    def initialize_swarm_router(self, agents: List[Agent], task: str):
        """Choose a swarm architecture for a pre-built agent roster and run it.

        Unlike :meth:`build_and_run_swarm`, this does not generate the
        agents themselves — it takes an existing ``agents`` list and asks
        the boss agent only to pick a ``swarm_type`` (and flow/rules) for
        them.

        Args:
            agents (List[Agent]): The agents to route the task through.
            task (str): The task to execute.

        Returns:
            Any: The result of the swarm router execution.

        Raises:
            Exception: If spec generation or router execution fails.
        """
        try:
            logger.info("Initializing swarm router")
            swarm_spec: SwarmRouterConfig = (
                self._generate_structured_output(
                    config=SwarmRouterConfig,
                    prompt=f"Create the swarm spec for the following task: {task}",
                )
            )
            self._validate_swarm_type(swarm_spec.swarm_type)

            logger.debug(
                f"Generated swarm spec: {swarm_spec.model_dump()}"
            )

            swarm_router = SwarmRouter(
                name=swarm_spec.name,
                description=swarm_spec.description,
                max_loops=self.max_loops,
                swarm_type=swarm_spec.swarm_type,
                rearrange_flow=swarm_spec.rearrange_flow,
                agents=agents,
                output_type="dict",
            )

            logger.info("Starting swarm router execution")
            return swarm_router.run(task)
        except Exception as e:
            logger.error(
                f"Error in swarm router initialization/execution: {str(e)}",
                exc_info=True,
            )
            raise

    def batch_run(self, tasks: List[str]):
        """Run the swarm builder on a list of tasks, sequentially.

        Args:
            tasks (List[str]): List of tasks to execute.

        Returns:
            List[Any]: List of results from each task execution.
        """
        return [self.run(task) for task in tasks]

    def create_agents_from_specs(
        self, agents_dictionary: Any
    ) -> List[Agent]:
        """Create agents from agent specifications.

        Args:
            agents_dictionary: Dictionary containing agent specifications
                (under an ``"agents"`` key), or an object exposing an
                ``.agents`` attribute (e.g. ``Agents``/``SwarmRouterConfig``).

        Returns:
            List[Agent]: List of created agents.

        Raises:
            ValueError: If any two agents share a non-empty ``agent_name``.

        Notes:
            - Handles both dict and Pydantic AgentSpec inputs
            - Maps 'description' field to 'agent_description' for Agent compatibility
        """
        # Handle both dict and object formats
        if isinstance(agents_dictionary, dict):
            agents_list = agents_dictionary.get("agents", [])
        else:
            agents_list = agents_dictionary.agents

        def _spec_name(item: Any) -> Optional[str]:
            if isinstance(item, dict):
                return item.get("agent_name")
            return getattr(item, "agent_name", None)

        self._check_duplicate_agent_names(
            [_spec_name(item) for item in agents_list]
        )

        agents = []
        for agent_config in agents_list:
            # Convert dict to AgentSpec if needed
            if isinstance(agent_config, dict):
                agent_config = AgentSpec(**agent_config)

            # Convert Pydantic model to dict for Agent initialization
            if isinstance(agent_config, BaseModel):
                agent_data = agent_config.model_dump()
            else:
                agent_data = agent_config

            # Handle parameter name mapping: description -> agent_description
            if (
                "description" in agent_data
                and "agent_description" not in agent_data
            ):
                agent_data["agent_description"] = agent_data.pop(
                    "description"
                )

            # Create agent from processed data
            agent = Agent(**agent_data)
            agents.append(agent)

        return agents

    def list_types(self) -> List[str]:
        """List all available execution types.

        Returns:
            List[str]: List of available execution types.
        """
        return swarm_types

    def run(
        self,
        task: str,
        execute: Optional[bool] = None,
        *args,
        **kwargs,
    ):
        """Run the swarm builder on a given task.

        Args:
            task (str): The task to execute.
            execute (bool, optional): When True, build real agents and a
                ``SwarmRouter`` from a freshly generated spec and run it
                (see :meth:`build_and_run_swarm`), returning a dict that
                includes the execution output. When False, only a spec-only
                result is returned, per ``swarm_type``. Defaults to
                ``self.auto_execute``.
            *args: Additional positional arguments (unused; accepted for
                call-site compatibility).
            **kwargs: Additional keyword arguments (unused; accepted for
                call-site compatibility).

        Returns:
            Any: A dict from :meth:`build_and_run_swarm` when executing;
                otherwise the result of ``create_agents``,
                ``create_router_config``, or agent objects, depending on
                ``swarm_type``.

        Raises:
            ValueError: If ``swarm_type`` is invalid.
            Exception: If spec generation or execution fails.
        """
        try:
            should_execute = (
                self.auto_execute if execute is None else execute
            )
            if should_execute:
                return self.build_and_run_swarm(task)

            if self.swarm_type == "return-agents":
                return self.create_agents(task)
            elif self.swarm_type == "return-swarm-router-config":
                return self.create_router_config(task)
            elif self.swarm_type == "return-agents-objects":
                agents = self.create_agents(task)
                return self.create_agents_from_specs(agents)
            else:
                raise ValueError(
                    f"Invalid execution type: {self.swarm_type}"
                )

        except Exception as e:
            logger.error(
                f"AutoSwarmBuilder: Error in swarm execution: {str(e)} Traceback: {traceback.format_exc()}",
                exc_info=True,
            )
            raise e
