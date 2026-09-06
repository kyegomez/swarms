import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.structs.omni_agent_types import AgentType
from swarms.utils.loguru_logger import initialize_logger
from swarms.utils.output_types import OutputType


logger = initialize_logger(log_folder="social_algorithms")


@dataclass
class SocialAlgorithmResult:
    """Result of executing a social algorithm."""

    algorithm_id: str
    execution_time: float
    total_steps: int
    successful_steps: int
    failed_steps: int
    communication_history: List[Dict[str, Any]]
    final_outputs: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


class SocialAlgorithmError(Exception):
    """Base exception for social algorithm errors."""

    pass


class InvalidAlgorithmError(SocialAlgorithmError):
    """Raised when an invalid algorithm is provided."""

    pass


class AgentNotFoundError(SocialAlgorithmError):
    """Raised when a required agent is not found."""

    pass


class SocialAlgorithms:
    """
    Run an arbitrary callable that decides how a group of agents talk to each other.

    The algorithm receives ``(agents, task, **kwargs)`` and may call the agents in
    any order. Every agent call made while it runs is recorded into
    ``self.conversation``, so the transcript is available without the algorithm
    having to report anything itself.

    Args:
        algorithm_id (str, optional): Unique identifier. Generated if omitted.
        name (str): Human-readable name for the algorithm.
        description (str): Description of what the algorithm does.
        agents (List[AgentType]): Agents that participate in the algorithm.
        social_algorithm (Callable): Callable defining the communication
            sequence. Must accept ``(agents, task, **kwargs)``.
        max_execution_time (float): Seconds allowed for execution.
        output_type (OutputType): Format of ``final_outputs``.
        verbose (bool): Whether to log progress.

    Attributes:
        conversation (Conversation): Transcript of every agent message, kept
            across runs.

    Raises:
        InvalidAlgorithmError: If ``social_algorithm`` is not callable.
        ValueError: If no agents are given, or ``max_execution_time`` is not positive.

    Example:
        >>> def pipeline(agents, task, **kwargs):
        ...     research = agents[0].run(f"Research: {task}")
        ...     return agents[1].run(f"Analyze: {research}")
        >>>
        >>> social_alg = SocialAlgorithms(
        ...     agents=[researcher, analyst],
        ...     social_algorithm=pipeline,
        ... )
        >>> result = social_alg.run("The impact of AI on healthcare")
        >>> print(social_alg.conversation.get_str())
    """

    def __init__(
        self,
        algorithm_id: str = None,
        name: str = "SocialAlgorithm",
        description: str = "A custom social algorithm for agent communication",
        agents: List[AgentType] = None,
        social_algorithm: Callable = None,
        max_execution_time: float = 300.0,
        output_type: OutputType = "dict",
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        self.algorithm_id = algorithm_id or str(uuid.uuid4())
        self.name = name
        self.description = description
        self.agents = agents or []
        self.social_algorithm = social_algorithm
        self.max_execution_time = max_execution_time
        self.output_type = output_type
        self.verbose = verbose

        self.execution_metadata: Dict[str, Any] = {}
        self.conversation = Conversation(
            name=f"{name}-Conversation", time_enabled=True
        )

        self._validate_inputs()

        if self.verbose:
            logger.info(
                f"Initialized {self.name} with {len(self.agents)} agents"
            )

    def _validate_inputs(self) -> None:
        """
        Validate the constructor inputs.

        Raises:
            InvalidAlgorithmError: If the social_algorithm is not callable.
            ValueError: If the agents list is empty or invalid.
        """
        if not self.agents:
            raise ValueError("At least one agent must be provided")

        if not all(isinstance(agent, Agent) for agent in self.agents):
            raise ValueError(
                "All agents must be instances of the Agent class"
            )

        if self.social_algorithm is not None and not callable(
            self.social_algorithm
        ):
            raise InvalidAlgorithmError(
                "social_algorithm must be callable"
            )

        if self.max_execution_time <= 0:
            raise ValueError("max_execution_time must be positive")

    def add_agent(self, agent: Agent) -> None:
        if not isinstance(agent, Agent):
            raise ValueError(
                "Agent must be an instance of the Agent class"
            )

        self.agents.append(agent)

        if self.verbose:
            logger.info(
                f"Added agent: {agent.agent_name} to {self.name}"
            )

    def remove_agent(self, agent_name: str) -> None:
        for index, agent in enumerate(self.agents):
            if agent.agent_name == agent_name:
                del self.agents[index]
                break
        else:
            raise AgentNotFoundError(
                f"No agent found with name: {agent_name}"
            )

        if self.verbose:
            logger.info(f"Removed agent: {agent_name}")

    def get_communication_history(self) -> List[Dict[str, Any]]:
        """
        Get the recorded agent messages.

        Returns:
            List[Dict[str, Any]]: The conversation messages, oldest first.
        """
        return list(self.conversation.conversation_history)

    def clear_communication_history(self) -> None:
        """Clear the recorded agent messages."""
        self.conversation.conversation_history.clear()

    def _log_communication(
        self,
        sender_agent: str,
        message: Any,
        receiver_agent: Optional[str] = None,
    ) -> None:
        """
        Record one agent message into the conversation.

        Args:
            sender_agent (str): Name of the agent that produced the message.
            message (Any): The message content.
            receiver_agent (str, optional): Name of the receiving agent, for
                agent-to-agent messages.
        """
        self.conversation.add(
            role=sender_agent,
            content=message,
            metadata=(
                {"receiver_agent": receiver_agent}
                if receiver_agent
                else None
            ),
        )

        if self.verbose:
            logger.info(
                f"{sender_agent} -> {receiver_agent or sender_agent}: {str(message)[:100]}"
            )

    @contextmanager
    def _recording_agents(self):
        """
        Route every agent's ``run`` and ``talk_to`` through the conversation log.

        Patches the agent instances rather than the ``Agent`` class, so agents
        outside this swarm and concurrent runs elsewhere are unaffected.

        Yields:
            None: For the duration of the algorithm call.
        """
        saved = [
            (
                agent,
                agent.__dict__.get("run"),
                agent.__dict__.get("talk_to"),
            )
            for agent in self.agents
        ]

        for agent in self.agents:
            agent.run = self._recorded_run(agent)
            agent.talk_to = self._recorded_talk_to(agent)

        try:
            yield
        finally:
            for agent, original_run, original_talk_to in saved:
                self._restore(agent, "run", original_run)
                self._restore(agent, "talk_to", original_talk_to)

    @staticmethod
    def _restore(agent: Agent, name: str, original: Any) -> None:
        """
        Undo one patched attribute on an agent.

        Args:
            agent (Agent): The agent to restore.
            name (str): Attribute name that was patched.
            original (Any): The prior instance attribute, or None if the agent
                had none and should fall back to the class method.
        """
        if original is None:
            agent.__dict__.pop(name, None)
        else:
            agent.__dict__[name] = original

    def _recorded_run(self, agent: Agent) -> Callable:
        """
        Build a ``run`` replacement that logs the agent's output.

        Args:
            agent (Agent): The agent being wrapped, before it is patched.

        Returns:
            Callable: A drop-in replacement for ``agent.run``.
        """
        # Whatever is bound now, so an instance override still runs
        original = agent.run

        def run(task, *args, **kwargs):
            result = original(task, *args, **kwargs)
            self._log_communication(agent.agent_name, result)
            return result

        return run

    def _recorded_talk_to(self, agent: Agent) -> Callable:
        """
        Build a ``talk_to`` replacement that logs the outgoing message.

        Args:
            agent (Agent): The agent being wrapped, before it is patched.

        Returns:
            Callable: A drop-in replacement for ``agent.talk_to``.
        """
        original = agent.talk_to

        def talk_to(other, task, *args, **kwargs):
            self._log_communication(
                agent.agent_name, task, other.agent_name
            )
            return original(other, task, *args, **kwargs)

        return talk_to

    def _execute_with_timeout(
        self, func: Callable, *args, **kwargs
    ) -> Any:
        """
        Execute a function with a timeout.

        Args:
            func (Callable): The function to execute.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            Any: The result of the function execution.

        Raises:
            TimeoutError: If the function execution exceeds max_execution_time.
        """
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError(
                f"Algorithm execution exceeded {self.max_execution_time} seconds"
            )

        # Set up timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(self.max_execution_time))

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Restore original handler
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    def _format_output(self, result: Any) -> Any:
        """
        Format the output according to the specified output_type.

        Args:
            result (Any): The raw result from the algorithm.

        Returns:
            Any: The formatted result.
        """
        if self.output_type == "dict" and not isinstance(
            result, dict
        ):
            return {"result": result}
        elif self.output_type == "list" and not isinstance(
            result, list
        ):
            return [result]
        elif self.output_type == "str" and not isinstance(
            result, str
        ):
            return str(result)

        return result

    def run(
        self,
        task: str,
        algorithm_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SocialAlgorithmResult:
        """
        Execute the social algorithm with the given task.

        Args:
            task (str): The task to execute using the social algorithm.
            algorithm_args (Dict[str, Any], optional): Additional arguments for
                the algorithm. Never mutated.
            **kwargs: Additional keyword arguments, which win on conflict.

        Returns:
            SocialAlgorithmResult: The result of executing the social algorithm.

        Raises:
            InvalidAlgorithmError: If no social algorithm is defined.
            TimeoutError: If execution exceeds max_execution_time.
            Exception: Whatever the algorithm raises.
        """
        if self.social_algorithm is None:
            raise InvalidAlgorithmError(
                "No social algorithm defined. Please provide a callable algorithm."
            )

        if self.verbose:
            logger.info(
                f"[{self.name}] Running {len(self.agents)} agents on: {task}"
            )

        self.conversation.add(role="User", content=task)
        opening_steps = len(self.conversation.conversation_history)
        algorithm_kwargs = {**(algorithm_args or {}), **kwargs}

        start_time = time.time()
        failed_steps = 0

        try:
            with self._recording_agents():
                if self.max_execution_time > 0:
                    result = self._execute_with_timeout(
                        self.social_algorithm,
                        self.agents,
                        task,
                        **algorithm_kwargs,
                    )
                else:
                    result = self.social_algorithm(
                        self.agents, task, **algorithm_kwargs
                    )
        except TimeoutError:
            logger.warning(
                f"[{self.name}] Timed out after {self.max_execution_time}s"
            )
            raise
        except Exception as e:
            logger.error(
                f"[{self.name}] Failed: {type(e).__name__}: {e}"
            )
            failed_steps = 1
            raise
        finally:
            execution_time = time.time() - start_time

        successful_steps = (
            len(self.conversation.conversation_history)
            - opening_steps
        )
        self.conversation.add(role=self.name, content=result)
        history = self.get_communication_history()

        if self.verbose:
            logger.info(
                f"[{self.name}] Completed in {execution_time:.2f}s over {successful_steps} messages"
            )

        return SocialAlgorithmResult(
            algorithm_id=self.algorithm_id,
            execution_time=execution_time,
            total_steps=successful_steps,
            successful_steps=successful_steps,
            failed_steps=failed_steps,
            communication_history=history,
            final_outputs=self._format_output(result),
            metadata=self.execution_metadata,
        )

    def get_algorithm_info(self) -> Dict[str, Any]:
        """
        Get information about the social algorithm.

        Returns:
            Dict[str, Any]: Information about the algorithm.
        """
        return {
            "algorithm_id": self.algorithm_id,
            "name": self.name,
            "description": self.description,
            "agent_count": len(self.agents),
            "agent_names": [
                agent.agent_name for agent in self.agents
            ],
            "has_algorithm": self.social_algorithm is not None,
            "max_execution_time": self.max_execution_time,
            "output_type": self.output_type,
            "verbose": self.verbose,
        }
