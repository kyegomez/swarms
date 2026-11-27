import asyncio
import contextlib
import copy
from concurrent.futures import as_completed
from typing import Any, Dict, List, Optional, get_args

from swarms.structs.comms_channel import (
    CommsChannel,
    claim_agents,
)
from swarms.structs.context_utils import (
    agent_answer,
    messages_for,
    split_last_turn,
)
from swarms.structs.conversation import Conversation
from swarms.structs.execution_utils import run_concurrently
from swarms.structs.ma_blocks import find_agent_by_name
from swarms.structs.serialization import SerializableMixin
from swarms.structs.usage_utils import total_usage
from swarms.telemetry.otel import (
    ContextThreadPoolExecutor,
    capture_init,
    trace_run,
)
from swarms.utils.any_to_str import any_to_str
from swarms.utils.generate_id import generate_id
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.loguru_logger import initialize_logger
from swarms.utils.output_types import OutputType
from swarms.utils.workspace_manager import WorkspaceManager

logger = initialize_logger(log_folder="rearrange")


def _split_steps(segments: List[str]) -> List[List[str]]:
    """Split arrow segments into their comma-separated agent names.

    Args:
        segments (List[str]): Flow segments, already split on the arrow.

    Returns:
        List[List[str]]: One inner list of agent names per segment.
    """
    return [
        [name.strip() for name in seg.split(",") if name.strip()]
        for seg in segments
    ]


class _SwarmMember:
    """
    A nested swarm that a flow runs like an agent.

    Args:
        swarm (Any): A structure with a name and a run method that takes a task.
    """

    def __init__(self, swarm: Any):
        name = getattr(swarm, "name", None)
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"{type(swarm).__name__} needs a name to appear in a flow"
            )
        self.swarm = swarm
        self.name = name
        self.agent_name = name

    def run(
        self,
        task: str = None,
        messages: Optional[List[Dict[str, str]]] = None,
        img: str = None,
        *args,
        **kwargs,
    ) -> str:
        """
        Run the swarm on the conversation so far, flattened into one task.

        Args:
            task (str): The newest turn of the shared conversation.
            messages (Optional[List[Dict[str, str]]]): The earlier turns.
            img (str): Optional image input.

        Returns:
            str: The swarm's output.
        """
        turns = [
            (
                f"{self.name}: {m['content']}"
                if m["role"] == "assistant"
                else m["content"]
            )
            for m in messages or []
        ]
        text = "\n\n".join([*turns, task or ""]).strip()
        if img is not None:
            kwargs["img"] = img
        output = self.swarm.run(text, *args, **kwargs)
        return (
            output if isinstance(output, str) else any_to_str(output)
        )

    async def arun_stream(
        self,
        task: str = None,
        messages: Optional[List[Dict[str, str]]] = None,
        img: str = None,
        **kwargs,
    ):
        """
        Yield the swarm's whole output as one chunk.

        Args:
            task (str): The newest turn of the shared conversation.
            messages (Optional[List[Dict[str, str]]]): The earlier turns.
            img (str): Optional image input.

        Returns:
            AsyncIterator[str]: A single chunk holding the output.
        """
        yield await asyncio.to_thread(
            self.run, task, messages, img, **kwargs
        )


def _as_member(member: Any) -> Any:
    """
    Wrap a swarm so the flow can run it, leaving agents as they are.

    Args:
        member (Any): An agent, or a swarm with a name and a run method.

    Returns:
        Any: The agent itself, or the wrapped swarm.
    """
    if hasattr(member, "agent_name"):
        return member
    return _SwarmMember(member)


# Derived from the Literal so it cannot drift from HistoryOutputType
_VALID_OUTPUT_TYPES = set(get_args(OutputType))


class AgentRearrange(SerializableMixin):
    """
    A sophisticated multi-agent system for task rearrangement and orchestration.

    The AgentRearrange class enables complex workflows where multiple agents can work
    sequentially or concurrently based on a defined flow pattern. It supports both
    sequential execution and concurrent execution within the same workflow.

    Key Features:
    - Sequential and concurrent agent execution
    - Nested swarms as steps of the flow, referenced by name
    - Custom flow patterns with arrow and comma syntax
    - Team awareness and sequential flow information
    - Memory system support
    - Batch and concurrent processing capabilities
    - Comprehensive error handling and logging

    Flow Syntax:
    - Use an arrow to define sequential execution
    - Use a comma to define concurrent execution
    - Combine both in one flow

    Attributes:
        id (str): Unique identifier for the agent rearrange system
        name (str): Human-readable name for the system
        description (str): Description of the system's purpose
        agents (List[Any]): The agents and nested swarms, in flow order
        flow (str): Flow pattern defining agent execution order
        max_loops (int): Maximum number of execution loops
        verbose (bool): Whether to enable verbose logging
        output_type (OutputType): Format for output results
        autosave (bool): Whether to save the conversation to the workspace
            directory after each run
        team_awareness (bool): Whether agents are aware of team structure
        time_enabled (bool): Whether to track timestamps
        message_id_on (bool): Whether to include message IDs
        conversation (Conversation): Conversation history management
    """

    def __init__(
        self,
        id: Optional[str] = None,
        name: str = "AgentRearrange",
        description: str = "A swarm of agents for rearranging tasks.",
        agents: List[Any] = None,
        flow: str = None,
        max_loops: int = 1,
        verbose: bool = False,
        output_type: OutputType = "all",
        autosave: bool = False,
        team_awareness: bool = False,
        time_enabled: bool = False,
        message_id_on: bool = False,
        collab_prompt: Optional[str] = None,
        channel: Optional[CommsChannel] = None,
    ):
        self.name = name
        self.description = description
        self.id = id or generate_id("agent-rearrange")
        self.agents = [_as_member(a) for a in agents or []]
        self.flow = flow if flow is not None else ""
        self.verbose = verbose
        self.max_loops = max_loops if max_loops > 0 else 1
        self.output_type = output_type
        self.autosave = autosave
        self.time_enabled = time_enabled
        self.message_id_on = message_id_on
        self.team_awareness = team_awareness
        self.collab_prompt = collab_prompt
        self.channel = channel
        # Seeds team awareness itself, seeding here too doubled the system message

        self.reliability_check()
        claim_agents(self.agents, channel, self.name)
        self._reset_conversation()

        self.workspace = WorkspaceManager(
            self,
            name=self.name or "agent-rearrange",
            verbose=self.verbose,
            enabled=self.autosave,
        )

        # Capture the full __init__ configuration if telemetry is enabled.
        capture_init(self)

    def reliability_check(self):
        """
        Validates the configuration parameters to ensure the system can run properly.

        Performs comprehensive validation checks on critical parameters including
        agents list, max_loops, flow pattern, and output_type to prevent runtime errors.
        """
        if self.agents is None or len(self.agents) == 0:
            raise ValueError("Agents list cannot be None or empty")

        self._require_unique_agent_names()

        if self.max_loops == 0:
            raise ValueError("max_loops cannot be 0")

        if self.flow is None or self.flow == "":
            raise ValueError("flow cannot be None or empty")

        if not all(self.steps):
            raise ValueError(f"flow {self.flow!r} has an empty step")

        if self.output_type is None or self.output_type == "":
            raise ValueError("output_type cannot be None or empty")

        if self.output_type not in _VALID_OUTPUT_TYPES:
            raise ValueError(
                f"output_type must be one of "
                f"{sorted(_VALID_OUTPUT_TYPES)}; got {self.output_type!r}"
            )

    def _require_unique_agent_names(self) -> None:
        """Reject two different members that share a name.

        Steps run by name, so a repeated name would run only the first
        member for every step that names it. The same member listed
        twice is allowed.

        Raises:
            ValueError: Two different members have the same name.
        """
        owners: Dict[str, Any] = {}

        for member in self.agents:
            # Each nested swarm gets a new wrapper, so compare what was passed
            source = (
                member.swarm
                if isinstance(member, _SwarmMember)
                else member
            )

            if (
                owners.setdefault(member.agent_name, source)
                is not source
            ):
                raise ValueError(
                    f"Two different agents are named '{member.agent_name}'. "
                    "AgentRearrange runs each step by name, so only the "
                    "first of them would run. Give each one its own name."
                )

    def set_custom_flow(self, flow: str):
        """
        Sets a custom flow pattern for agent execution.

        Allows dynamic modification of the execution flow after initialization.
        The flow pattern defines how agents should be executed in sequence or
        parallel using an arrow for sequential execution and a comma for
        concurrent execution.

        The new flow is validated immediately so typos surface at the call site
        rather than at the next run.

        Args:
            flow (str): The new flow pattern to use for agent execution.
        """
        previous_flow = self.flow
        self.flow = flow
        try:
            self.validate_flow()
        except ValueError:
            self.flow = previous_flow
            raise
        self._log("info", f"Custom flow set: {flow}")

    def explain(self, return_str: bool = False) -> Optional[str]:
        """Print or return the resolved execution plan for this flow.

        Walks the flow string and lists every step in order, marking each
        step as sequential or parallel. Does NOT invoke any agents or LLMs.
        Useful for CI smoke tests, debugging, and pre-flight verification.

        Args:
            return_str: When True, returns the plan as a string. When False
                (default), prints it and returns None.

        Returns:
            The plan string if return_str is True; otherwise None.
        """
        self.validate_flow()

        total_invocations = 0
        rows: List[tuple] = []
        for i, agent_names in enumerate(self.steps, start=1):
            total_invocations += len(agent_names)
            label = ", ".join(agent_names)
            if len(agent_names) == 1:
                kind = "[sequential]"
            else:
                kind = f"[parallel, {len(agent_names)} agents]"
            rows.append((i, label, kind))

        width = max((len(label) for _, label, _ in rows), default=0)
        lines: List[str] = [f"Flow: {self.flow}", ""]
        for i, label, kind in rows:
            lines.append(f"Step {i}: {label:<{width}}  {kind}")
        lines.append("")
        lines.append(
            f"{len(self.steps)} steps, {total_invocations} agent invocations "
            f"across {self.max_loops} loop(s)."
        )

        plan = "\n".join(lines)
        if return_str:
            return plan
        print(plan)
        return None

    def add_agent(self, agent: Any):
        """
        Adds an agent, or a nested swarm, to the swarm.

        Args:
            agent (Any): The agent or swarm to be added.
        """
        agent = _as_member(agent)
        self._log(
            "info", f"Adding agent {agent.agent_name} to the swarm."
        )
        self.agents.append(agent)

    def remove_agent(self, agent_name: str):
        """
        Removes an agent from the swarm.

        Args:
            agent_name (str): The name of the agent to be removed.
        """
        for index, agent in enumerate(self.agents):
            if agent.agent_name == agent_name:
                self._log(
                    "info",
                    f"Removing agent {agent_name} from the swarm.",
                )
                del self.agents[index]
                return
        raise ValueError(
            f"No agent named {agent_name!r} in the swarm."
        )

    def add_agents(self, agents: List[Any]):
        """
        Adds multiple agents, or nested swarms, to the swarm.

        Args:
            agents (List[Any]): The agents or swarms to be added.
        """
        self.agents.extend(_as_member(a) for a in agents)

    @property
    def steps(self) -> List[List[str]]:
        """The flow as steps, each a list of agent names that run together.

        Each arrow separates a step, and names inside a step run together.
        Cached against the flow string, so reassigning flow (as
        set_custom_flow does) re-parses on next access without an explicit
        invalidation call.

        Returns:
            List[List[str]]: One inner list per arrow step.
        """
        if getattr(self, "_steps_for", None) != self.flow:
            self._steps = _split_steps((self.flow or "").split("->"))
            self._steps_for = self.flow
        return self._steps

    def validate_flow(self):
        """
        Validates the flow pattern.

        A valid flow is a non-empty string built from one or more steps
        separated by an arrow (sequential dependence). Each step is one or
        more comma-separated agent names that execute concurrently.
        Pure-concurrent flows and pure-sequential flows are both valid.

        Returns:
            bool: True if the flow pattern is valid.
        """
        if not self.flow or not self.flow.strip():
            raise ValueError("flow cannot be empty")

        for raw_step, agent_names in zip(
            self.flow.split("->"), self.steps
        ):
            if not agent_names:
                raise ValueError(
                    f"Empty agent name in flow segment {raw_step!r}"
                )
            for agent_name in agent_names:
                try:
                    find_agent_by_name(self.agents, agent_name)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"Agent '{agent_name}' is not registered."
                    )

        self._log("info", f"Flow: {self.flow} is valid.")
        return True

    def _get_sequential_awareness(
        self,
        agent_name: str,
        steps: List[List[str]],
        task_idx: int = None,
    ) -> str:
        """
        Determines the sequential awareness information for an agent in a sequential flow.

        Args:
            agent_name (str): The name of the current agent.
            steps (List[List[str]]): The parsed flow, one list of agent names
                per step. Passed in rather than read from self.steps so a
                custom_tasks run reflects the spliced flow it executes.
            task_idx (int, optional): The exact position index of this agent invocation
                in the flow. When provided, uses this directly instead of searching by name.
                This is essential for repeated agents so each occurrence gets
                correct positional awareness.

        Returns:
            str: A string describing the agents ahead and behind in the sequence.
        """
        if task_idx is not None:
            position = task_idx
        else:
            position = next(
                (
                    i
                    for i, names in enumerate(steps)
                    if agent_name in names
                ),
                None,
            )

        if position is None:
            return ""

        parts = []
        if position > 0:
            parts.append(
                f"Agent ahead: {', '.join(steps[position - 1])}"
            )
        if position < len(steps) - 1:
            parts.append(
                f"Agent behind: {', '.join(steps[position + 1])}"
            )

        if parts:
            return f"Sequential awareness: {' | '.join(parts)}"
        return ""

    def _get_sequential_flow_info(self) -> str:
        """
        Gets information about the overall sequential flow structure.

        Returns:
            str: A string describing the sequential flow structure.
        """
        if not self.flow or "->" not in self.flow:
            return ""

        steps = self.steps
        flow_info = []

        for i, agent_names in enumerate(steps):
            if not agent_names:
                continue
            line = f"Step {i + 1}: {', '.join(agent_names)}"
            if i > 0:
                line += f" (follows: {', '.join(steps[i - 1])})"
            if i < len(steps) - 1:
                line += f" (leads to: {', '.join(steps[i + 1])})"
            flow_info.append(line)

        if flow_info:
            return "Sequential Flow Structure:\n" + "\n".join(
                flow_info
            )
        return ""

    def get_agent_sequential_awareness(self, agent_name: str) -> str:
        """
        Gets the sequential awareness information for a specific agent.

        Args:
            agent_name (str): The name of the agent to get awareness for.

        Returns:
            str: A string describing the agents ahead and behind in the sequence.
        """
        if not self.flow or "->" not in self.flow:
            return ""

        return self._get_sequential_awareness(agent_name, self.steps)

    def get_sequential_flow_structure(self) -> str:
        """
        Gets the overall sequential flow structure information.

        Returns:
            str: A string describing the complete sequential flow structure.
        """
        return self._get_sequential_flow_info()

    def _reset_conversation(self) -> None:
        """Start a fresh shared conversation, re-seeding team awareness."""
        self.conversation = Conversation(
            time_enabled=self.time_enabled,
            token_count=False,
            message_id_on=self.message_id_on,
            channel=self.channel,
        )

        if self.team_awareness is True:
            sequential_info = self._get_sequential_flow_info()
            if sequential_info:
                self.conversation.add("system", sequential_info)

    def _messages_for(
        self, agent_name: str, extra_system: Optional[str] = None
    ) -> tuple:
        """
        The shared conversation as typed turns for one agent.

        Args:
            agent_name: The agent about to run.
            extra_system: Per-agent guidance - collaboration rules, flow
                position - prepended as a system turn. It is not written to
                the shared conversation, so other agents never see it.

        Returns:
            A tuple of prior messages and task: the conversation prefix as
            chat messages, and the newest turn as this run's task.
        """
        prior, task = split_last_turn(
            messages_for(agent_name, self.conversation)
        )

        preamble = "\n\n".join(
            part
            for part in (self.collab_prompt, extra_system)
            if part
        )
        if preamble:
            prior = [{"role": "system", "content": preamble}] + prior

        return prior, task

    def _run_concurrent_workflow(
        self,
        agent_names: List[str],
        img: str = None,
        *args,
        **kwargs,
    ) -> Dict[str, str]:
        """
        Executes agents concurrently when comma is detected in the flow.

        This method handles the parallel execution of multiple agents when they
        are separated by commas in the flow pattern. All specified agents run
        simultaneously and their results are collected and returned.

        Args:
            agent_names (List[str]): List of agent names to run concurrently.
                These agents will execute in parallel.
            img (str, optional): Image input for agents that support it.
                Defaults to None.
            *args: Additional positional arguments passed to agent execution.
            **kwargs: Additional keyword arguments passed to agent execution.

        Returns:
            Dict[str, str]: Dictionary mapping agent names to their execution results.
                Keys are agent names, values are their respective outputs.
        """
        self._log(
            "info", f"Running agents in parallel: {agent_names}"
        )

        agents_to_run = []
        missing = []
        for name in agent_names:
            try:
                agents_to_run.append(
                    find_agent_by_name(self.agents, name)
                )
            except (TypeError, ValueError):
                missing.append(name)
        if missing:
            raise ValueError(
                f"Agent(s) {missing} not registered in this AgentRearrange instance."
            )

        # Built per agent, not shared: each sees its own turns as `assistant`.
        results = [None] * len(agents_to_run)
        with ContextThreadPoolExecutor(
            max_workers=len(agents_to_run)
        ) as executor:
            future_to_index = {}
            for index, (agent, name) in enumerate(
                zip(agents_to_run, agent_names)
            ):
                prior, step_task = self._messages_for(name)
                future = executor.submit(
                    agent.run,
                    task=step_task,
                    messages=prior,
                    img=img,
                    **kwargs,
                )
                future_to_index[future] = index

            for future in as_completed(future_to_index):
                results[future_to_index[future]] = future.result()

        response_dict = {}
        for i, agent_name in enumerate(agent_names):
            result = results[i]

            # run() returns the agent's whole conversation by default, record the answer
            result = agent_answer(agents_to_run[i], fallback=result)

            self.conversation.add(agent_name, result)
            response_dict[agent_name] = result
            self._log("debug", f"Agent {agent_name} output: {result}")

        return response_dict

    def _run_sequential_workflow(
        self,
        agent_name: str,
        steps: List[List[str]],
        task_idx: int = None,
        img: str = None,
        *args,
        **kwargs,
    ) -> str:
        """
        Executes a single agent sequentially.

        This method handles the sequential execution of a single agent in the flow.
        It provides sequential awareness information to the agent if team_awareness
        is enabled, allowing the agent to understand its position in the workflow.

        Args:
            agent_name (str): Name of the agent to run sequentially.
            tasks (List[str]): List of all tasks in the flow for awareness context.
                Used to determine the agent's position and provide awareness info.
            task_idx (int, optional): The position index of this agent in the flow.
                Essential for repeated agents so each occurrence gets correct awareness.
            img (str, optional): Image input for agents that support it.
                Defaults to None.
            *args: Additional positional arguments passed to agent execution.
            **kwargs: Additional keyword arguments passed to agent execution.

        Returns:
            str: The result from the agent's execution, converted to string format.
        """
        self._log("info", f"Running agent sequentially: {agent_name}")

        try:
            agent = find_agent_by_name(self.agents, agent_name)
        except (TypeError, ValueError):
            raise ValueError(
                f"Agent '{agent_name}' is not registered in this AgentRearrange instance."
            )

        # The system_prompt is baked into the LLM at build time, so assigning to it here never reaches the request.
        awareness_info = (
            self._get_sequential_awareness(
                agent_name, steps, task_idx=task_idx
            )
            if self.team_awareness
            else None
        )
        if awareness_info:
            self._log(
                "info",
                f"Added sequential awareness for {agent_name}: {awareness_info}",
            )

        prior, step_task = self._messages_for(
            agent_name, extra_system=awareness_info
        )
        current_task = agent.run(
            task=step_task,
            messages=prior,
            img=img,
            *args,
            **kwargs,
        )

        if not isinstance(current_task, str):
            current_task = any_to_str(current_task)

        # Record the answer: run() honours output_type, which defaults to the whole transcript.
        current_task = agent_answer(agent, fallback=current_task)

        self.conversation.add(agent.agent_name, current_task)

        return current_task

    def _run(
        self,
        task: str = None,
        img: str = None,
        custom_tasks: Dict[str, str] = None,
        *args,
        **kwargs,
    ):
        """
        Runs the swarm to rearrange the tasks according to the defined flow.

        This is the core execution method that orchestrates the entire workflow.
        It processes the flow pattern, executes agents sequentially or concurrently
        as specified, and returns the results in the requested format.

        Args:
            task (str, optional): The initial task to be processed by the swarm.
                This is added to the conversation history. Defaults to None.
            img (str, optional): Image input for agents that support it.
                Defaults to None.
            custom_tasks (Dict[str, str], optional): Custom tasks for specific agents.
                Allows overriding the main task for specific agents in the flow.
                Defaults to None.
            *args: Additional positional arguments passed to agent execution.
            **kwargs: Additional keyword arguments passed to agent execution.

        Returns:
            Union[str, List[str], Dict[str, str]]: The processed output in the format
                specified by output_type:
                - "all": String containing all agent responses concatenated
                - "final": Only the final agent's response
                - "list": List of all agent responses
                - "dict": Dict mapping agent names to their responses
        """
        # A reused instance would otherwise carry the previous task's transcript
        self._reset_conversation()

        self.conversation.add("User", task)

        # A raw copy, custom_tasks splices into it and self.steps stays unmutated
        tasks = self.flow.split("->")
        response_dict = {}

        self._log(
            "info", f"Starting task execution with {len(tasks)} steps"
        )

        # Handle custom tasks
        if custom_tasks is not None:
            self._log("info", "Processing custom tasks")
            c_agent_name, c_task = next(iter(custom_tasks.items()))
            position = tasks.index(c_agent_name)

            if position > 0:
                tasks[position - 1] += "->" + c_task
            else:
                tasks.insert(position, c_task)

        loop_count = 0
        while loop_count < self.max_loops:
            self._log(
                "info",
                f"Starting loop {loop_count + 1}/{self.max_loops}",
            )

            steps = _split_steps(tasks)

            for task_idx, agent_names in enumerate(steps):

                if len(agent_names) > 1:
                    # Concurrent processing - comma detected
                    concurrent_results = (
                        self._run_concurrent_workflow(
                            agent_names=agent_names,
                            img=img,
                            *args,
                            **kwargs,
                        )
                    )
                    response_dict.update(concurrent_results)

                else:
                    # Sequential processing
                    agent_name = agent_names[0]
                    result = self._run_sequential_workflow(
                        agent_name=agent_name,
                        steps=steps,
                        task_idx=task_idx,
                        img=img,
                        *args,
                        **kwargs,
                    )

                    # Use indexed key to preserve all outputs
                    if agent_name in response_dict:
                        response_dict[f"{agent_name}_{task_idx}"] = (
                            result
                        )
                    else:
                        response_dict[agent_name] = result

            loop_count += 1

        self._log("info", "Task execution completed")

        return history_output_formatter(
            conversation=self.conversation,
            type=self.output_type,
        )

    def _catch_error(self, e: Exception):
        """Log an error from the swarm run and re-raise it."""
        self._log(
            "error",
            f"AgentRearrange: Id: {self.id}, Name: {self.name}. An error occurred with your agent '{self.name}': Error: {e}. Traceback: {e.__traceback__}",
        )

        raise e

    @trace_run(
        "AgentRearrange.run",
        input_params=("task", "img"),
    )
    def run(
        self,
        task: str = None,
        img: str = None,
        *args,
        **kwargs,
    ):
        """
        Execute the agent rearrangement task with comprehensive logging and error handling.

        This is the main public method for executing tasks through the agent rearrange
        system. It provides telemetry logging, error handling, and delegates to the
        internal _run method for actual execution.

        Args:
            task (str, optional): The task to execute through the agent workflow.
                Defaults to None.
            img (str, optional): Path to input image if required by any agents.
                Defaults to None.
            *args: Additional positional arguments passed to the internal _run() method.
            **kwargs: Additional keyword arguments passed to the internal _run() method.

        Returns:
            The result from executing the task through the agent rearrange system.
            The format depends on the configured output_type.

        """
        try:
            out = self._run(
                task=task,
                img=img,
                *args,
                **kwargs,
            )

            return out

        except Exception as e:
            self._catch_error(e)
        finally:
            self.workspace.save_conversation(self.conversation)

    def __call__(self, task: str, *args, **kwargs):
        """
        Make the class callable by executing the run() method.

        Enables the AgentRearrange instance to be called directly as a function,
        providing a convenient interface for task execution.
        """
        return self.run(task=task, *args, **kwargs)

    def _clone_for_task(self) -> "AgentRearrange":
        """Build an isolated clone of this orchestrator for one task.

        Each clone gets:
        - A fresh Conversation so per-task history does not bleed across
          parallel workers and the original conversation is never mutated.
        - A per-agent deepcopy where the agent supports it (so stateful
          agents do not race), falling back to the original instance when
          deepcopy fails (for example the agent holds a thread lock, an HTTP
          connection pool, or other non-picklable resources).

        This replaces a previous full deepcopy of the orchestrator that
        raised TypeError whenever any agent contained non-picklable internals.
        """
        clone = self.__class__.__new__(self.__class__)
        clone.__dict__.update(self.__dict__)
        clone.workspace = WorkspaceManager(clone, enabled=False)

        clone.conversation = Conversation(
            time_enabled=self.time_enabled,
            token_count=False,
            message_id_on=self.message_id_on,
            channel=self.channel,
        )

        cloned_agents: List[Any] = []
        for agent in self.agents:
            try:
                cloned_agents.append(copy.deepcopy(agent))
            except Exception:
                cloned_agents.append(agent)
        clone.agents = cloned_agents

        return clone

    def batch_run(
        self,
        tasks: List[str],
        img: Optional[List[str]] = None,
        batch_size: int = 10,
        *args,
        **kwargs,
    ) -> List[str]:
        """
        Process multiple tasks in batches for efficient execution.

        This method allows processing multiple tasks by dividing them into
        smaller batches and processing each batch sequentially. This is useful
        for managing memory usage and resource allocation when dealing with
        large numbers of tasks.

        Args:
            tasks (List[str]): List of tasks to process through the agent workflow.
            img (Optional[List[str]]): Optional list of images corresponding to tasks.
                Must be the same length as tasks list. Defaults to None.
            batch_size (int): Number of tasks to process simultaneously in each batch.
                Defaults to 10.
            *args: Additional positional arguments passed to individual task execution.
            **kwargs: Additional keyword arguments passed to individual task execution.

        Returns:
            List[str]: List of results corresponding to input tasks in the same order.
        """
        if batch_size <= 0:
            raise ValueError(
                f"batch_size must be a positive integer, got {batch_size}"
            )
        if img is not None and len(img) != len(tasks):
            raise ValueError(
                f"img length ({len(img)}) must match tasks length ({len(tasks)})"
            )
        results = []
        for i in range(0, len(tasks), batch_size):
            batch_tasks = tasks[i : i + batch_size]
            batch_imgs = (
                img[i : i + batch_size]
                if img
                else [None] * len(batch_tasks)
            )

            # Each task gets an isolated clone; _clone_for_task survives agents a full deepcopy(self) would choke on.
            max_workers = len(batch_tasks)
            futures_ordered = []
            with ContextThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                for task_item, img_path in zip(
                    batch_tasks, batch_imgs
                ):
                    clone = self._clone_for_task()
                    future = executor.submit(
                        clone.run,
                        task_item,
                        img_path,
                        *args,
                        **kwargs,
                    )
                    futures_ordered.append(future)
                batch_results = [f.result() for f in futures_ordered]
            results.extend(batch_results)

        return results

    def concurrent_run(
        self,
        tasks: List[str],
        img: Optional[List[str]] = None,
        max_workers: Optional[int] = None,
        *args,
        **kwargs,
    ) -> List[str]:
        """
        Process multiple tasks concurrently using ThreadPoolExecutor.

        This method enables true parallel processing of multiple tasks by using
        Python's ThreadPoolExecutor to run tasks simultaneously across multiple
        threads. This is ideal for I/O-bound tasks or when you want maximum
        parallelization.

        Args:
            tasks (List[str]): List of tasks to process through the agent workflow.
            img (Optional[List[str]]): Optional list of images corresponding to tasks.
                Must be the same length as tasks list. Defaults to None.
            max_workers (Optional[int]): Maximum number of worker threads to use.
                If None, uses the default ThreadPoolExecutor behavior. Defaults to None.
            *args: Additional positional arguments passed to individual task execution.
            **kwargs: Additional keyword arguments passed to individual task execution.

        Returns:
            List[str]: List of results corresponding to input tasks in the same order.
        """

        def run_isolated(task, *run_args, **run_kwargs):
            return self._clone_for_task().run(
                task, *run_args, **run_kwargs
            )

        return run_concurrently(
            run_isolated,
            tasks,
            *args,
            img=img,
            max_workers=max_workers,
            **kwargs,
        )

    async def run_async(
        self,
        task: str,
        img: Optional[str] = None,
        *args,
        **kwargs,
    ) -> Any:
        """
        Asynchronously executes a task through the agent workflow.

        This method enables asynchronous execution of tasks by running the
        synchronous run method in a separate thread using asyncio.to_thread.
        This is ideal for integrating the agent workflow into async applications
        or when you want non-blocking execution.

        Args:
            task (str): The task to be executed through the agent workflow.
            img (Optional[str]): Optional image input for the task. Defaults to None.
            *args: Additional positional arguments passed to the run method.
            **kwargs: Additional keyword arguments passed to the run method.

        Returns:
            Any: The result of the task execution, format depends on output_type setting.
        """

        try:
            return await asyncio.to_thread(
                self.run, task=task, img=img, *args, **kwargs
            )
        except Exception as e:
            self._catch_error(e)

    async def _run_sequential_workflow_stream(
        self,
        agent_name: str,
        steps: List[List[str]],
        task_idx: int = None,
        img: str = None,
        with_events: bool = False,
        **kwargs,
    ):
        """Stream a single sequential agent.

        Yields agent name and token pairs by default, or structured event
        dicts when with_events is True.
        """
        agent = find_agent_by_name(self.agents, agent_name)

        # Injected via a temporary system-prompt extension, see _run_sequential_workflow
        awareness_info = self._get_sequential_awareness(
            agent_name, steps, task_idx=task_idx
        )
        original_system_prompt = getattr(agent, "system_prompt", None)
        if awareness_info and original_system_prompt is not None:
            agent.system_prompt = (
                f"{original_system_prompt}\n\n{awareness_info}"
            )

        if with_events:
            yield {"type": "agent_start", "agent": agent_name}

        chunks: List[str] = []
        prior, step_task = self._messages_for(agent_name)
        run_kwargs = {"task": step_task, "messages": prior}
        if img is not None:
            run_kwargs["img"] = img
        run_kwargs.update(kwargs)
        try:
            async for token in agent.arun_stream(**run_kwargs):
                chunks.append(token)
                if with_events:
                    yield {
                        "type": "token",
                        "agent": agent_name,
                        "token": token,
                    }
                else:
                    yield (agent_name, token)
        finally:
            if awareness_info and original_system_prompt is not None:
                agent.system_prompt = original_system_prompt

        output = "".join(chunks)
        self.conversation.add(agent.agent_name, output)
        if with_events:
            yield {
                "type": "agent_end",
                "agent": agent_name,
                "output": output,
            }

    async def _run_concurrent_workflow_stream(
        self,
        agent_names: List[str],
        img: str = None,
        with_events: bool = False,
        **kwargs,
    ):
        """Run agents in parallel, interleaving their tokens.

        Yielding control after each token forces a scheduler hop so
        concurrent producers get fair time on the event loop, which
        interleaves tokens from each agent instead of draining one
        agent fully before the next.
        """
        q: asyncio.Queue = asyncio.Queue()
        DONE = object()
        ERROR = object()
        base_inputs = {
            name: self._messages_for(name) for name in agent_names
        }
        results: Dict[str, List[str]] = {
            name: [] for name in agent_names
        }

        if with_events:
            for name in agent_names:
                yield {"type": "agent_start", "agent": name}

        async def producer(name: str):
            try:
                agent = find_agent_by_name(self.agents, name)
                prior, step_task = base_inputs[name]
                run_kwargs = {
                    "task": step_task,
                    "messages": prior,
                }
                if img is not None:
                    run_kwargs["img"] = img
                run_kwargs.update(kwargs)
                async for token in agent.arun_stream(**run_kwargs):
                    results[name].append(token)
                    await q.put((name, token))
            except Exception as e:  # noqa: BLE001
                await q.put((ERROR, e))
            finally:
                await q.put((DONE, name))

        producer_tasks = [
            asyncio.create_task(producer(name))
            for name in agent_names
        ]

        try:
            finished = 0
            while finished < len(agent_names):
                kind, payload = await q.get()
                if kind is DONE:
                    finished += 1
                    if with_events:
                        finished_name = payload
                        yield {
                            "type": "agent_end",
                            "agent": finished_name,
                            "output": "".join(results[finished_name]),
                        }
                    continue
                if kind is ERROR:
                    for t in producer_tasks:
                        t.cancel()
                    raise payload
                if with_events:
                    yield {
                        "type": "token",
                        "agent": kind,
                        "token": payload,
                    }
                else:
                    yield (kind, payload)
                # Yield so the consumer does not drain one agent's full burst before the scheduler hops.
                await asyncio.sleep(0)
        finally:
            await asyncio.gather(
                *producer_tasks, return_exceptions=True
            )

        for name in agent_names:
            self.conversation.add(name, "".join(results[name]))

    async def arun_stream(
        self,
        task: str = None,
        img: str = None,
        with_events: bool = False,
        **kwargs,
    ):
        """Stream tokens from each agent in flow order.

        - Sequential segments stream tokens one agent at a time.
        - Parallel segments interleave tokens from concurrent agents
          fairly by yielding control between tokens.

        Args:
            task: Initial task fed into the first agent in the flow.
            img: Optional image input forwarded to every agent.
            with_events: When False (default), yield agent name and token
                pairs. When True, yield structured event dicts for agent
                start, each token, and agent end.

        Not yet supported in streaming mode: max_loops greater than 1,
        and custom_tasks. Use run for those.
        """
        # A reused instance would otherwise carry the previous task's transcript
        self._reset_conversation()

        self.conversation.add("User", task)

        for task_idx, agent_names in enumerate(self.steps):

            if len(agent_names) > 1:
                async for evt in self._run_concurrent_workflow_stream(
                    agent_names=agent_names,
                    img=img,
                    with_events=with_events,
                    **kwargs,
                ):
                    yield evt
            else:
                async for evt in self._run_sequential_workflow_stream(
                    agent_name=agent_names[0],
                    steps=self.steps,
                    task_idx=task_idx,
                    img=img,
                    with_events=with_events,
                    **kwargs,
                ):
                    yield evt

    def run_stream(
        self,
        task: str = None,
        img: str = None,
        with_events: bool = False,
        **kwargs,
    ):
        """Sync generator version of arun_stream.

        Bridges the async generator to a sync iterator using a thread and queue.
        Use arun_stream directly when you already have a running event
        loop (for example inside a FastAPI handler).
        """
        import queue as _queue
        import threading

        q: _queue.Queue = _queue.Queue()
        DONE = object()
        exc_holder: List[Optional[Exception]] = [None]
        consumer: List[asyncio.Task] = []

        def runner():
            async def consume():
                consumer.append(asyncio.current_task())
                async for evt in self.arun_stream(
                    task=task,
                    img=img,
                    with_events=with_events,
                    **kwargs,
                ):
                    q.put(evt)

            try:
                asyncio.run(consume())
            except asyncio.CancelledError:
                pass
            except Exception as e:  # noqa: BLE001
                exc_holder[0] = e
            finally:
                q.put(DONE)

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        finished = False
        try:
            while True:
                item = q.get()
                if item is DONE:
                    finished = True
                    break
                yield item
        finally:
            if not finished:
                run = consumer[0]
                with contextlib.suppress(RuntimeError):
                    run.get_loop().call_soon_threadsafe(run.cancel)
        t.join()
        if exc_holder[0] is not None:
            raise exc_holder[0]

    @property
    def usage(self) -> dict:
        """
        Provider token usage summed over every agent this structure holds.

        Returns:
            A new dict with input_tokens, output_tokens, cached_tokens,
            reasoning_tokens and total_tokens, totalled across every run.
        """
        return total_usage(self)
