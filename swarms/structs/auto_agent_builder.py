import json
from typing import Any, Dict, List, Optional, Union

from swarms.prompts.auto_agent_builder_prompt import (
    AUTO_AGENT_BUILDER_SYSTEM_PROMPT,
)
from swarms.structs.agent import Agent
from swarms.structs.execution_utils import batched_run
from swarms.telemetry.otel import capture_init, trace_run
from swarms.utils.loguru_logger import initialize_logger

logger = initialize_logger(log_folder="auto_agent_builder")

DEFAULT_MODEL_NAME = "gpt-5.4"

# Kept as a module-level alias so existing imports of BUILDER_SYSTEM_PROMPT keep
# working now that the text lives in swarms/prompts/.
BUILDER_SYSTEM_PROMPT = AUTO_AGENT_BUILDER_SYSTEM_PROMPT

# The builder is forced to call this, so the provider enforces the shape and
# there is no free-form JSON to parse out of a prose response.
BUILD_AGENTS_TOOL = {
    "type": "function",
    "function": {
        "name": "build_agents",
        "description": (
            "Return the roster of agents needed to accomplish the task. "
            "Call this exactly once with every agent in the team."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "agents": {
                    "type": "array",
                    "description": "The agents that make up the team.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Unique, short, descriptive agent name, e.g. 'Churn-Analyst'.",
                            },
                            "description": {
                                "type": "string",
                                "description": "One sentence on what this agent is responsible for.",
                            },
                            "system_prompt": {
                                "type": "string",
                                "description": (
                                    "The agent's persona and instructions, written in the second "
                                    "person and specific to this task."
                                ),
                            },
                            "model_name": {
                                "type": "string",
                                "description": (
                                    "LiteLLM model string for this agent, e.g. 'gpt-5.4', "
                                    "'gpt-5.4-mini', 'claude-sonnet-4-6', 'groq/llama-3.3-70b-versatile'."
                                ),
                            },
                        },
                        "required": [
                            "name",
                            "description",
                            "system_prompt",
                            "model_name",
                        ],
                    },
                }
            },
            "required": ["agents"],
        },
    },
}

REQUIRED_FIELDS = (
    "name",
    "description",
    "system_prompt",
    "model_name",
)


def _extract_agents(tool_output: Any) -> List[Dict[str, str]]:
    """Parse the roster out of a forced ``build_agents`` tool call.

    Providers return tool calls in two shapes: a plain dict
    ``{"function": {"name", "arguments"}}`` (the MCP-normalized form), or a raw
    provider object such as litellm's ``ChatCompletionMessageToolCall`` where
    ``function`` and ``arguments`` are *attributes* rather than keys. Both are
    normalized here so either parses identically.

    Args:
        tool_output: Raw provider output from the forced tool invocation.

    Returns:
        List[Dict[str, str]]: One dict per agent, each carrying exactly
        ``name``, ``description``, ``system_prompt`` and ``model_name``.
        Malformed or missing output yields an empty list rather than raising —
        the caller decides whether an empty roster is fatal.
    """
    if isinstance(tool_output, list):
        tool_output = tool_output[0] if tool_output else None
    if not tool_output:
        return []

    if not isinstance(tool_output, dict) and hasattr(
        tool_output, "model_dump"
    ):
        try:
            tool_output = tool_output.model_dump()
        except Exception:
            pass

    if isinstance(tool_output, dict):
        fn = tool_output.get("function")
    else:
        fn = getattr(tool_output, "function", None)
    if not fn:
        return []

    args = (
        fn.get("arguments")
        if isinstance(fn, dict)
        else getattr(fn, "arguments", None)
    )
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            logger.error(
                "build_agents returned arguments that are not valid JSON"
            )
            return []
    if not isinstance(args, dict):
        return []

    raw = args.get("agents")
    if not isinstance(raw, list):
        return []

    agents: List[Dict[str, str]] = []
    seen = set()
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        # Drop partial entries rather than constructing an Agent with a missing
        # system_prompt, which would silently fall back to the default persona.
        if any(not entry.get(f) for f in REQUIRED_FIELDS):
            logger.warning(
                f"Skipping incomplete agent config: {entry!r}"
            )
            continue
        config = {f: str(entry[f]).strip() for f in REQUIRED_FIELDS}
        # Agent memory is keyed on agent_name, so duplicates would share and
        # corrupt each other's state.
        if config["name"] in seen:
            logger.warning(
                f"Skipping duplicate agent name: {config['name']!r}"
            )
            continue
        seen.add(config["name"])
        agents.append(config)

    return agents


class AutoAgentBuilder:
    """Generate a roster of agents for a task.

    A single builder agent is forced to call ``build_agents`` and answer with a
    roster. What :meth:`run` hands back is controlled by ``return_dict``:

    - ``return_dict=False`` (default) — constructed :class:`Agent` objects,
      ready to run or pass to a multi-agent structure.
    - ``return_dict=True`` — the raw configuration dicts, for inspection,
      serialization, or editing before you construct anything.

    :meth:`build_agents` and :meth:`build_configs` ignore the flag and always
    return agents and dicts respectively, for when a caller wants one specific
    shape regardless of how the builder was configured.

    Attributes:
        name (str): Name of this builder instance.
        description (str): What this builder is for.
        model_name (str): Model backing the builder agent itself. This is not
            the model given to the generated agents — the builder chooses those
            per agent.
        max_agents (int): Upper bound on roster size. The limit is stated to the
            model and enforced on the returned list, since a model may ignore it.
        system_prompt (str): Instructions for the builder agent.
        agent_kwargs (dict): Extra keyword arguments forwarded to every
            generated ``Agent``, e.g. ``max_loops`` or ``streaming_on``.
        return_dict (bool): Whether :meth:`run` returns configuration dicts
            instead of constructed agents.
        verbose (bool): Whether to log the generated roster.

    Example:
        >>> # Configurations only
        >>> builder = AutoAgentBuilder(max_agents=3, return_dict=True)
        >>> configs = builder.run("Research and summarize the EV market.")
        >>> [c["name"] for c in configs]
        ['Market-Researcher', 'Data-Analyst', 'Report-Writer']

        >>> # Live agents
        >>> agents = AutoAgentBuilder(max_agents=3).run("Same task.")
        >>> [a.agent_name for a in agents]
        ['Market-Researcher', 'Data-Analyst', 'Report-Writer']
    """

    def __init__(
        self,
        name: str = "auto-agent-builder",
        description: str = "Generates agent configurations from a task",
        model_name: str = DEFAULT_MODEL_NAME,
        max_agents: int = 5,
        num_agents: Optional[int] = None,
        system_prompt: str = BUILDER_SYSTEM_PROMPT,
        agent_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = False,
        verbose: bool = False,
    ):
        """Initialize the builder.

        Args:
            name (str): Name of this builder instance.
            description (str): What this builder is for.
            model_name (str): Model backing the builder agent.
            max_agents (int): Upper bound on roster size. This is a *ceiling*,
                not a target — the builder is instructed to prefer the smallest
                roster that covers the task, so it will routinely return fewer.
                Use ``num_agents`` to demand a specific count. Must be >= 1.
            num_agents (Optional[int]): Exact number of agents to generate. When
                set, this overrides ``max_agents`` and the builder's
                prefer-fewer guidance, and the count is verified on the result.
                Must be >= 1.
            system_prompt (str): Instructions for the builder agent.
            agent_kwargs (Optional[Dict[str, Any]]): Extra keyword arguments
                forwarded to every generated ``Agent``. Keys that collide with
                the four generated fields are ignored. Unused when
                ``return_dict`` is True, since no agent is constructed.
            return_dict (bool): When True, :meth:`run` returns the raw
                configuration dicts instead of constructed ``Agent`` objects.
                Use this to inspect, serialize, or edit the roster before
                building anything from it.
            verbose (bool): Whether to log the generated roster.

        Raises:
            ValueError: If ``max_agents`` or ``num_agents`` is less than 1.
        """
        if max_agents < 1:
            raise ValueError("max_agents must be at least 1.")
        if num_agents is not None and num_agents < 1:
            raise ValueError("num_agents must be at least 1.")

        self.name = name
        self.description = description
        self.model_name = model_name
        self.max_agents = max_agents
        self.num_agents = num_agents
        self.system_prompt = system_prompt
        self.agent_kwargs = agent_kwargs or {}
        self.return_dict = return_dict
        self.verbose = verbose

        capture_init(self)

    def _builder_agent(self) -> Agent:
        """Build the single agent that produces the roster.

        Returns:
            Agent: A one-loop agent carrying the forced ``build_agents`` schema.
        """
        return Agent(
            agent_name=self.name,
            agent_description=self.description,
            system_prompt=self.system_prompt,
            model_name=self.model_name,
            max_loops=1,
            # The roster depends only on the task, so carrying state across runs
            # would let an earlier task's team leak into a later one.
            output_type="final",
            persistent_memory=False,
            temperature=None,
            top_p=None,
            print_on=False,
            tools_list_dictionary=[BUILD_AGENTS_TOOL],
            **self.agent_kwargs,
        )

    @trace_run("AutoAgentBuilder.build_configs")
    def build_configs(self, task: str) -> List[Dict[str, str]]:
        """Generate agent configurations for a task, always as dicts.

        Ignores ``return_dict`` — this is the shape-specific entry point for
        callers that want configurations regardless of how the builder was
        configured.

        Args:
            task (str): The task the generated team should be able to handle.

        Returns:
            List[Dict[str, str]]: One dict per agent with ``name``,
            ``description``, ``system_prompt`` and ``model_name``. Truncated to
            ``max_agents``.

        Raises:
            ValueError: If ``task`` is empty, or the builder returns no usable
                agent configurations.
        """
        if not task or not task.strip():
            raise ValueError("task must be a non-empty string.")

        if self.num_agents is not None:
            # Stated twice, and as a hard requirement, because the system
            # prompt's prefer-fewer guidance otherwise pulls the count down.
            instruction = (
                f"Design exactly {self.num_agents} agents for this task — not "
                f"fewer, not more. This exact count is a hard requirement and "
                f"overrides your usual preference for a smaller roster. Split "
                f"the work along a finer seam if you need to reach "
                f"{self.num_agents}, but give every agent a real job. Then "
                f"call build_agents with all {self.num_agents}."
            )
            limit = self.num_agents
        else:
            instruction = (
                f"Design at most {self.max_agents} agents for this task, then "
                f"call build_agents with the roster. Use the smallest roster "
                f"that fully covers the task."
            )
            limit = self.max_agents

        prompt = f"Task:\n{task}\n\n{instruction}"

        output = self._builder_agent().run(prompt)
        agents = _extract_agents(output)

        if not agents:
            raise ValueError(
                "The builder agent did not return any usable agent "
                "configurations. Check that the model supports function "
                "calling and try rephrasing the task."
            )

        # The model is told the limit but may exceed it; enforce it here.
        if len(agents) > limit:
            logger.warning(
                f"Builder returned {len(agents)} agents, truncating to {limit}."
            )
            agents = agents[:limit]

        # An exact count is a request, not something we can fabricate — surface
        # the shortfall rather than silently handing back a smaller roster.
        if (
            self.num_agents is not None
            and len(agents) < self.num_agents
        ):
            logger.warning(
                f"Requested exactly {self.num_agents} agents but the builder "
                f"returned {len(agents)}. Try a task with more separable parts, "
                f"or lower num_agents."
            )

        if self.verbose:
            for a in agents:
                logger.info(
                    f"{a['name']} ({a['model_name']}): {a['description']}"
                )

        return agents

    def build_agents(self, task: str) -> List[Agent]:
        """Generate configurations and construct the ``Agent`` objects.

        Ignores ``return_dict`` — this is the shape-specific entry point for
        callers that want agents regardless of how the builder was configured.

        Args:
            task (str): The task the generated team should be able to handle.

        Returns:
            List[Agent]: Constructed agents, ready to run or hand to a
            multi-agent structure.

        Raises:
            ValueError: Propagated from :meth:`build_configs` on an empty task
                or an empty roster.
        """
        return [
            Agent(
                agent_name=config["name"],
                agent_description=config["description"],
                system_prompt=config["system_prompt"],
                model_name=config["model_name"],
                **self.agent_kwargs,
            )
            for config in self.build_configs(task)
        ]

    def run(
        self, task: str
    ) -> Union[List[Agent], List[Dict[str, str]]]:
        """Generate the roster in the shape this builder was configured for.

        Args:
            task (str): The task the generated team should be able to handle.

        Returns:
            Union[List[Agent], List[Dict[str, str]]]: Configuration dicts when
            ``return_dict`` is True, otherwise constructed ``Agent`` objects.

        Raises:
            ValueError: Propagated from :meth:`build_configs` on an empty task
                or an empty roster.
        """
        if self.return_dict:
            return self.build_configs(task)
        return self.build_agents(task)

    def __call__(
        self, task: str
    ) -> Union[List[Agent], List[Dict[str, str]]]:
        """Alias for :meth:`run`.

        Args:
            task (str): The task the generated team should be able to handle.

        Returns:
            Union[List[Agent], List[Dict[str, str]]]: Whatever :meth:`run`
            returns for this builder's ``return_dict`` setting.
        """
        return self.run(task)

    def batch_run(self, tasks: List[str]) -> Any:
        """Generate the roster for each task in sequence.

        Args:
            tasks (List[str]): The tasks the generated team should be able to handle.

        Returns:
            List[Union[List[Agent], List[Dict[str, str]]]]: The rosters for each task.
        """
        return batched_run(self.run, tasks)
