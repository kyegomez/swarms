import time
import uuid
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass

from swarms.structs.agent import Agent
from swarms.structs.omni_agent_types import AgentType
from swarms.utils.loguru_logger import initialize_logger
from swarms.utils.output_types import OutputType

logger = initialize_logger(log_folder="social_algorithms")


@dataclass
class CommunicationStep:
    """Represents a single step in a social algorithm."""

    step_id: str
    sender_agent: str
    receiver_agent: str
    message: str
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SocialAlgorithmResult:
    """Result of executing a social algorithm."""

    algorithm_id: str
    execution_time: float
    total_steps: int
    successful_steps: int
    failed_steps: int
    communication_history: List[CommunicationStep]
    final_outputs: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


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
    A flexible framework for defining and executing custom social algorithms
    that control how agents communicate and interact with each other.

    This class allows users to upload any arbitrary social algorithm as a callable
    that defines the sequence of communication between agents. The algorithm should
    specify how agents talk to each other, in what order, and what information
    should be shared.

    Attributes:
        algorithm_id (str): Unique identifier for the algorithm instance.
        name (str): Human-readable name for the algorithm.
        description (str): Description of what the algorithm does.
        agents (List[AgentType]): List of agents that will participate in the algorithm.
        social_algorithm (Callable): The callable that defines the communication sequence.
        max_execution_time (float): Maximum time allowed for algorithm execution.
        output_type (OutputType): Format of the output from the algorithm.
        verbose (bool): Whether to enable verbose logging.

    Example:
        >>> from swarms import Agent, SocialAlgorithms
        >>>
        >>> # Define a custom social algorithm
        >>> def custom_communication_algorithm(agents, task, **kwargs):
        ...     # Agent 1 researches the topic
        ...     research_result = agents[0].run(f"Research: {task}")
        ...
        ...     # Agent 2 analyzes the research
        ...     analysis = agents[1].run(f"Analyze this research: {research_result}")
        ...
        ...     # Agent 3 synthesizes the findings
        ...     synthesis = agents[2].run(f"Synthesize: {research_result} + {analysis}")
        ...
        ...     return {
        ...         "research": research_result,
        ...         "analysis": analysis,
        ...         "synthesis": synthesis
        ...     }
        >>>
        >>> # Create agents
        >>> researcher = Agent(agent_name="Researcher", model_name="gpt-4o-mini")
        >>> analyst = Agent(agent_name="Analyst", model_name="gpt-4o-mini")
        >>> synthesizer = Agent(agent_name="Synthesizer", model_name="gpt-4o-mini")
        >>>
        >>> # Create social algorithm
        >>> social_alg = SocialAlgorithms(
        ...     name="Research-Analysis-Synthesis",
        ...     agents=[researcher, analyst, synthesizer],
        ...     social_algorithm=custom_communication_algorithm,
        ...     verbose=True
        ... )
        >>>
        >>> # Run the algorithm
        >>> result = social_alg.run("The impact of AI on healthcare")
        >>> print(result)
    """

    def __init__(
        self,
        algorithm_id: str = None,
        name: str = "SocialAlgorithm",
        description: str = "A custom social algorithm for agent communication",
        agents: List[AgentType] = None,
        social_algorithm: Callable = None,
        max_execution_time: float = 300.0,  # 5 minutes default
        output_type: OutputType = "dict",
        verbose: bool = False,
        enable_communication_logging: bool = False,
        parallel_execution: bool = False,
        max_workers: int = None,
        *args,
        **kwargs,
    ):
        """
        Initialize a SocialAlgorithms instance.

        Args:
            algorithm_id (str, optional): Unique identifier for the algorithm.
                If None, a UUID will be generated.
            name (str): Human-readable name for the algorithm.
            description (str): Description of what the algorithm does.
            agents (List[AgentType]): List of agents that will participate in the algorithm.
            social_algorithm (Callable): The callable that defines the communication sequence.
                Must accept (agents, task, **kwargs) as parameters.
            max_execution_time (float): Maximum time allowed for algorithm execution in seconds.
            output_type (OutputType): Format of the output from the algorithm.
            verbose (bool): Whether to enable verbose logging.
            enable_communication_logging (bool): Whether to log communication steps.
            parallel_execution (bool): Whether to enable parallel execution where possible.
            max_workers (int, optional): Maximum number of workers for parallel execution.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Raises:
            InvalidAlgorithmError: If the social_algorithm is not callable or invalid.
            ValueError: If agents list is empty or invalid.
        """
        self.algorithm_id = algorithm_id or str(uuid.uuid4())
        self.name = name
        self.description = description
        self.agents = agents or []
        self.social_algorithm = social_algorithm
        self.max_execution_time = max_execution_time
        self.output_type = output_type
        self.verbose = verbose
        self.enable_communication_logging = (
            enable_communication_logging
        )
        self.parallel_execution = parallel_execution
        self.max_workers = max_workers

        # Communication tracking
        self.communication_history: List[CommunicationStep] = []
        self.execution_metadata: Dict[str, Any] = {}

        # Validate inputs
        self._validate_inputs()

        # Initialize agent mapping for quick lookup
        self._agent_map = {
            agent.agent_name: agent for agent in self.agents
        }

        if self.verbose:
            logger.info(
                f"Initialized SocialAlgorithm: {self.name} with {len(self.agents)} agents"
            )

    def _validate_inputs(self) -> None:
        """
        Validate the inputs provided to the SocialAlgorithms constructor.

        Raises:
            InvalidAlgorithmError: If the social_algorithm is not callable.
            ValueError: If agents list is empty or invalid.
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

    def _log_communication(
        self,
        sender_agent: str,
        receiver_agent: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a communication step between agents.

        Args:
            sender_agent (str): Name of the sending agent.
            receiver_agent (str): Name of the receiving agent.
            message (str): The message being sent.
            metadata (Dict[str, Any], optional): Additional metadata about the communication.
        """
        if not self.enable_communication_logging:
            return

        step = CommunicationStep(
            step_id=str(uuid.uuid4()),
            sender_agent=sender_agent,
            receiver_agent=receiver_agent,
            message=message,
            timestamp=time.time(),
            metadata=metadata,
        )

        self.communication_history.append(step)

        if self.verbose:
            logger.info(
                f"Communication: {sender_agent} -> {receiver_agent}: {message[:100]}..."
            )

    def _log_execution_step(
        self,
        step: str,
        details: Optional[Dict[str, Any]] = None,
        level: str = "info",
    ) -> None:
        """
        Log a key execution step with optional details.

        Args:
            step (str): Description of the execution step.
            details (Dict[str, Any], optional): Additional details about the step.
            level (str): Log level ('info', 'debug', 'warning', 'error').
        """
        if not self.verbose:
            return

        message = f"[{self.name}] {step}"
        if details:
            message += f" - Details: {details}"

        if level == "debug":
            logger.debug(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        else:
            logger.info(message)

    def _get_agent_by_name(self, agent_name: str) -> Agent:
        """
        Get an agent by its name.

        Args:
            agent_name (str): Name of the agent to retrieve.

        Returns:
            Agent: The agent with the specified name.

        Raises:
            AgentNotFoundError: If no agent with the given name is found.
        """
        if agent_name not in self._agent_map:
            raise AgentNotFoundError(
                f"Agent '{agent_name}' not found in the algorithm"
            )

        return self._agent_map[agent_name]

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

    def add_agent(self, agent: Agent) -> None:
        """
        Add an agent to the social algorithm.

        Args:
            agent (Agent): The agent to add.
        """
        if not isinstance(agent, Agent):
            raise ValueError(
                "Agent must be an instance of the Agent class"
            )

        self.agents.append(agent)
        self._agent_map[agent.agent_name] = agent

        if self.verbose:
            logger.info(f"Added agent: {agent.agent_name}")

    def remove_agent(self, agent_name: str) -> None:
        """
        Remove an agent from the social algorithm.

        Args:
            agent_name (str): Name of the agent to remove.

        Raises:
            AgentNotFoundError: If no agent with the given name is found.
        """
        if agent_name not in self._agent_map:
            raise AgentNotFoundError(
                f"Agent '{agent_name}' not found"
            )

        # Remove from both lists
        self.agents = [
            agent
            for agent in self.agents
            if agent.agent_name != agent_name
        ]
        del self._agent_map[agent_name]

        if self.verbose:
            logger.info(f"Removed agent: {agent_name}")

    def get_agent_names(self) -> List[str]:
        """
        Get a list of all agent names in the algorithm.

        Returns:
            List[str]: List of agent names.
        """
        return list(self._agent_map.keys())

    def get_communication_history(self) -> List[CommunicationStep]:
        """
        Get the communication history for this algorithm execution.

        Returns:
            List[CommunicationStep]: List of communication steps.
        """
        return self.communication_history.copy()

    def clear_communication_history(self) -> None:
        """Clear the communication history."""
        self.communication_history.clear()

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
            algorithm_args (Dict[str, Any], optional): Additional arguments for the algorithm.
            **kwargs: Additional keyword arguments.

        Returns:
            SocialAlgorithmResult: The result of executing the social algorithm.

        Raises:
            InvalidAlgorithmError: If no social algorithm is defined.
            TimeoutError: If the algorithm execution exceeds max_execution_time.
            Exception: If the algorithm execution fails.
        """
        if self.social_algorithm is None:
            raise InvalidAlgorithmError(
                "No social algorithm defined. Please provide a callable algorithm."
            )

        self._log_execution_step(
            "Starting social algorithm execution",
            {
                "algorithm_name": self.name,
                "task": task,
                "agent_count": len(self.agents),
                "agent_names": [
                    agent.agent_name for agent in self.agents
                ],
                "max_execution_time": self.max_execution_time,
            },
        )

        # Clear previous communication history
        self._log_execution_step(
            "Clearing previous communication history"
        )
        self.clear_communication_history()

        # Prepare algorithm arguments
        self._log_execution_step(
            "Preparing algorithm arguments",
            {
                "algorithm_args": algorithm_args,
                "additional_kwargs": kwargs,
            },
        )
        algorithm_kwargs = algorithm_args or {}
        algorithm_kwargs.update(kwargs)

        # Add communication logging wrapper if enabled
        if self.enable_communication_logging:
            self._log_execution_step(
                "Wrapping algorithm with communication logging"
            )
            wrapped_algorithm = self._wrap_algorithm_with_logging()
            wrapped_algorithm.social_algorithms_instance = self
        else:
            self._log_execution_step(
                "Using algorithm without communication logging"
            )
            wrapped_algorithm = self.social_algorithm

        start_time = time.time()
        successful_steps = 0
        failed_steps = 0

        try:
            # Execute the algorithm with timeout
            if self.max_execution_time > 0:
                self._log_execution_step(
                    "Executing algorithm with timeout",
                    {"timeout_seconds": self.max_execution_time},
                )
                result = self._execute_with_timeout(
                    wrapped_algorithm,
                    self.agents,
                    task,
                    **algorithm_kwargs,
                )
            else:
                self._log_execution_step(
                    "Executing algorithm without timeout"
                )
                result = wrapped_algorithm(
                    self.agents, task, **algorithm_kwargs
                )

            successful_steps = len(self.communication_history)
            self._log_execution_step(
                "Algorithm execution completed successfully",
                {
                    "successful_steps": successful_steps,
                    "communication_steps": len(
                        self.communication_history
                    ),
                },
            )

        except TimeoutError:
            self._log_execution_step(
                "Algorithm execution timed out",
                {"timeout_seconds": self.max_execution_time},
                level="error",
            )
            raise
        except Exception as e:
            self._log_execution_step(
                "Algorithm execution failed",
                {"error": str(e), "error_type": type(e).__name__},
                level="error",
            )
            failed_steps = 1
            raise
        finally:
            execution_time = time.time() - start_time

        # Format the output
        self._log_execution_step(
            "Formatting output", {"output_type": self.output_type}
        )
        formatted_result = self._format_output(result)

        # Create result object
        self._log_execution_step("Creating algorithm result object")
        algorithm_result = SocialAlgorithmResult(
            algorithm_id=self.algorithm_id,
            execution_time=execution_time,
            total_steps=len(self.communication_history),
            successful_steps=successful_steps,
            failed_steps=failed_steps,
            communication_history=self.communication_history.copy(),
            final_outputs=formatted_result,
            metadata=self.execution_metadata,
        )

        self._log_execution_step(
            "Algorithm execution completed",
            {
                "execution_time": f"{execution_time:.2f} seconds",
                "total_communication_steps": len(
                    self.communication_history
                ),
                "successful_steps": successful_steps,
                "failed_steps": failed_steps,
            },
        )

        return algorithm_result

    def _wrap_algorithm_with_logging(self) -> Callable:
        """
        Wrap the social algorithm with communication logging.

        Returns:
            Callable: The wrapped algorithm with logging capabilities.
        """

        def wrapped_algorithm(agents, task, **kwargs):
            # Store original agent methods
            original_talk_to = Agent.talk_to
            original_run = Agent.run

            def logged_talk_to(
                agent_self, agent, task, img=None, *args, **kwargs
            ):
                # Log the communication
                self._log_communication(
                    agent_self.agent_name, agent.agent_name, task
                )
                # Call original method
                return original_talk_to(
                    agent_self, agent, task, img, *args, **kwargs
                )

            def logged_run(
                agent_self, task, img=None, *args, **kwargs
            ):
                # Log the communication (self-communication)
                self._log_communication(
                    agent_self.agent_name, agent_self.agent_name, task
                )
                # Call original method
                return original_run(
                    agent_self, task, img, *args, **kwargs
                )

            # Temporarily replace methods
            Agent.talk_to = logged_talk_to
            Agent.run = logged_run

            try:
                # Execute the algorithm
                result = self.social_algorithm(agents, task, **kwargs)
                return result
            finally:
                # Restore original methods
                Agent.talk_to = original_talk_to
                Agent.run = original_run

        return wrapped_algorithm

    def run_async(
        self,
        task: str,
        algorithm_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SocialAlgorithmResult:
        """
        Execute the social algorithm asynchronously.

        Args:
            task (str): The task to execute using the social algorithm.
            algorithm_args (Dict[str, Any], optional): Additional arguments for the algorithm.
            **kwargs: Additional keyword arguments.

        Returns:
            SocialAlgorithmResult: The result of executing the social algorithm.
        """
        import asyncio

        async def async_execution():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, self.run, task, algorithm_args, **kwargs
            )

        return asyncio.run(async_execution())

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
            "enable_communication_logging": self.enable_communication_logging,
            "parallel_execution": self.parallel_execution,
            "max_workers": self.max_workers,
        }

    def __str__(self) -> str:
        """String representation of the SocialAlgorithms instance."""
        return f"SocialAlgorithms(name='{self.name}', agents={len(self.agents)}, algorithm_id='{self.algorithm_id}')"

    def __repr__(self) -> str:
        """Detailed string representation of the SocialAlgorithms instance."""
        return f"SocialAlgorithms(algorithm_id='{self.algorithm_id}', name='{self.name}', agents={self.agents})"
