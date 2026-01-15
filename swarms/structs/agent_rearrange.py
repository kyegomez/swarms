import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Union
import asyncio
from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.structs.multi_agent_exec import run_agents_concurrently
from swarms.structs.swarm_id import swarm_id
from swarms.telemetry.main import log_agent_data
from swarms.utils.any_to_str import any_to_str
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.loguru_logger import initialize_logger
from swarms.utils.output_types import OutputType

logger = initialize_logger(log_folder="rearrange")


class AgentRearrange:
    """
    A sophisticated multi-agent system for task rearrangement and orchestration.

    The AgentRearrange class enables complex workflows where multiple agents can work
    sequentially or concurrently based on a defined flow pattern. It supports both
    sequential execution (using '->') and concurrent execution (using ',') within
    the same workflow.

    Key Features:
    - Sequential and concurrent agent execution
    - Custom flow patterns with arrow (->) and comma (,) syntax
    - Team awareness and sequential flow information
    - Human-in-the-loop integration
    - Memory system support
    - Batch and concurrent processing capabilities
    - Comprehensive error handling and logging

    Flow Syntax:
    - Use '->' to define sequential execution: "agent1 -> agent2 -> agent3"
    - Use ',' to define concurrent execution: "agent1, agent2 -> agent3"
    - Combine both: "agent1 -> agent2, agent3 -> agent4"
    - Use 'H' for human-in-the-loop: "agent1 -> H -> agent2"

    Attributes:
        id (str): Unique identifier for the agent rearrange system
        name (str): Human-readable name for the system
        description (str): Description of the system's purpose
        agents (Dict[str, Agent]): Dictionary mapping agent names to Agent objects
        flow (str): Flow pattern defining agent execution order
        max_loops (int): Maximum number of execution loops
        verbose (bool): Whether to enable verbose logging
        memory_system (Any): Optional memory system for persistence
        human_in_the_loop (bool): Whether to enable human interaction
        custom_human_in_the_loop (Callable): Custom human interaction handler
        output_type (OutputType): Format for output results
        autosave (bool): Whether to automatically save execution data
        rules (str): System rules and constraints
        team_awareness (bool): Whether agents are aware of team structure
        time_enabled (bool): Whether to track timestamps
        message_id_on (bool): Whether to include message IDs
        conversation (Conversation): Conversation history management

    Example:
        >>> from swarms import Agent, AgentRearrange
        >>>
        >>> # Create agents
        >>> agent1 = Agent(name="researcher", ...)
        >>> agent2 = Agent(name="writer", ...)
        >>> agent3 = Agent(name="reviewer", ...)
        >>>
        >>> # Define flow: agent1 runs first, then agent2 and agent3 run concurrently
        >>> flow = "researcher -> writer, reviewer"
        >>>
        >>> # Create rearrange system
        >>> rearrange_system = AgentRearrange(
        ...     agents=[agent1, agent2, agent3],
        ...     flow=flow,
        ...     max_loops=1,
        ...     team_awareness=True
        ... )
        >>>
        >>> # Execute task
        >>> result = rearrange_system.run("Research and write a report")
    """

    def __init__(
        self,
        id: str = swarm_id(),
        name: str = "AgentRearrange",
        description: str = "A swarm of agents for rearranging tasks.",
        agents: List[Union[Agent, Callable]] = None,
        flow: str = None,
        max_loops: int = 1,
        verbose: bool = True,
        memory_system: Any = None,
        human_in_the_loop: bool = False,
        custom_human_in_the_loop: Optional[
            Callable[[str], str]
        ] = None,
        output_type: OutputType = "all",
        autosave: bool = True,
        rules: str = None,
        team_awareness: bool = False,
        time_enabled: bool = False,
        message_id_on: bool = False,
    ):
        """
        Initialize the AgentRearrange system.

        Args:
            id (str): Unique identifier for the agent rearrange system.
                Defaults to a generated swarm ID.
            name (str): Human-readable name for the system.
                Defaults to "AgentRearrange".
            description (str): Description of the system's purpose.
                Defaults to "A swarm of agents for rearranging tasks.".
            agents (List[Union[Agent, Callable]], optional): List of agents to include
                in the system. Can be Agent objects or callable functions.
                Defaults to None.
            flow (str, optional): Flow pattern defining agent execution order.
                Uses '->' for sequential and ',' for concurrent execution.
                Defaults to None.
            max_loops (int): Maximum number of execution loops. Must be > 0.
                Defaults to 1.
            verbose (bool): Whether to enable verbose logging.
                Defaults to True.
            memory_system (Any, optional): Optional memory system for persistence.
                Defaults to None.
            human_in_the_loop (bool): Whether to enable human interaction points.
                Defaults to False.
            custom_human_in_the_loop (Callable[[str], str], optional): Custom function
                for handling human interaction. Takes input string, returns response.
                Defaults to None.
            output_type (OutputType): Format for output results. Can be "all", "final",
                "list", or "dict". Defaults to "all".
            autosave (bool): Whether to automatically save execution data.
                Defaults to True.
            rules (str, optional): System rules and constraints to add to conversation.
                Defaults to None.
            team_awareness (bool): Whether agents should be aware of team structure
                and sequential flow. Defaults to False.
            time_enabled (bool): Whether to track timestamps in conversations.
                Defaults to False.
            message_id_on (bool): Whether to include message IDs in conversations.
                Defaults to False.

        Raises:
            ValueError: If agents list is None or empty, max_loops is 0,
                flow is None or empty, or output_type is None or empty.

        Note:
            The agents parameter is converted to a dictionary mapping agent names
            to Agent objects for efficient lookup during execution.
        """
        self.name = name
        self.description = description
        self.id = id
        self.agents = {agent.agent_name: agent for agent in agents}
        self.flow = flow if flow is not None else ""
        self.verbose = verbose
        self.max_loops = max_loops if max_loops > 0 else 1
        self.memory_system = memory_system
        self.human_in_the_loop = human_in_the_loop
        self.custom_human_in_the_loop = custom_human_in_the_loop
        self.output_type = output_type
        self.autosave = autosave
        self.time_enabled = time_enabled
        self.message_id_on = message_id_on

        self.conversation = Conversation(
            name=f"{self.name}-Conversation",
            time_enabled=self.time_enabled,
            token_count=False,
            message_id_on=self.message_id_on,
        )

        if rules:
            self.conversation.add("user", rules)

        if team_awareness is True:
            # agents_info = get_agents_info(agents=self.agents, team_name=self.name)

            # Add sequential flow information if available
            sequential_info = self._get_sequential_flow_info()
            if sequential_info:
                # agents_info += "\n\n" + sequential_info
                self.conversation.add("system", sequential_info)

            # self.conversation.add("system", agents_info)

        self.reliability_check()

    def reliability_check(self):
        """
        Validates the configuration parameters to ensure the system can run properly.

        Performs comprehensive validation checks on critical parameters including
        agents list, max_loops, flow pattern, and output_type to prevent runtime errors.

        Raises:
            ValueError: If any of the following conditions are met:
                - agents list is None or empty
                - max_loops is 0
                - flow is None or empty string
                - output_type is None or empty string

        Note:
            This method is called automatically during initialization to ensure
            the system is properly configured before execution.
        """
        if self.agents is None or len(self.agents) == 0:
            raise ValueError("Agents list cannot be None or empty")

        if self.max_loops == 0:
            raise ValueError("max_loops cannot be 0")

        if self.flow is None or self.flow == "":
            raise ValueError("flow cannot be None or empty")

        if self.output_type is None or self.output_type == "":
            raise ValueError("output_type cannot be None or empty")

    def set_custom_flow(self, flow: str):
        """
        Sets a custom flow pattern for agent execution.

        Allows dynamic modification of the execution flow after initialization.
        The flow pattern defines how agents should be executed in sequence or
        parallel using the standard syntax ('->' for sequential, ',' for concurrent).

        Args:
            flow (str): The new flow pattern to use for agent execution.
                Must follow the syntax: "agent1 -> agent2, agent3 -> agent4"

        Note:
            The flow will be validated on the next execution. If invalid,
            a ValueError will be raised during the run() method.

        Example:
            >>> rearrange_system.set_custom_flow("researcher -> writer, editor")
        """
        self.flow = flow
        logger.info(f"Custom flow set: {flow}")

    def add_agent(self, agent: Agent):
        """
        Adds an agent to the swarm.

        Args:
            agent (Agent): The agent to be added.
        """
        logger.info(f"Adding agent {agent.agent_name} to the swarm.")
        self.agents[agent.agent_name] = agent

    def track_history(
        self,
        agent_name: str,
        result: str,
    ):
        """
        Tracks the execution history for a specific agent.

        Records the result of an agent's execution in the swarm history
        for later analysis or debugging purposes.

        Args:
            agent_name (str): The name of the agent whose result to track.
            result (str): The result/output from the agent's execution.

        Note:
            This method is typically called internally during agent execution
            to maintain a complete history of all agent activities.
        """
        self.swarm_history[agent_name].append(result)

    def remove_agent(self, agent_name: str):
        """
        Removes an agent from the swarm.

        Args:
            agent_name (str): The name of the agent to be removed.
        """
        del self.agents[agent_name]

    def add_agents(self, agents: List[Agent]):
        """
        Adds multiple agents to the swarm.

        Args:
            agents (List[Agent]): A list of Agent objects.
        """
        for agent in agents:
            self.agents[agent.agent_name] = agent

    def validate_flow(self):
        """
        Validates the flow pattern.

        Raises:
            ValueError: If the flow pattern is incorrectly formatted or contains duplicate agent names.

        Returns:
            bool: True if the flow pattern is valid.
        """
        if "->" not in self.flow:
            raise ValueError(
                "Flow must include '->' to denote the direction of the task."
            )

        agents_in_flow = []

        # Arrow
        tasks = self.flow.split("->")

        # For the task in tasks
        for task in tasks:
            agent_names = [name.strip() for name in task.split(",")]

            # Loop over the agent names
            for agent_name in agent_names:
                if (
                    agent_name not in self.agents
                    and agent_name != "H"
                ):
                    raise ValueError(
                        f"Agent '{agent_name}' is not registered."
                    )
                agents_in_flow.append(agent_name)

        # # If the length of the agents does not equal the length of the agents in flow
        # if len(set(agents_in_flow)) != len(agents_in_flow):
        #     raise ValueError(
        #         "Duplicate agent names in the flow are not allowed."
        #     )

        logger.info(f"Flow: {self.flow} is valid.")
        return True

    def _get_sequential_awareness(
        self, agent_name: str, tasks: List[str]
    ) -> str:
        """
        Determines the sequential awareness information for an agent in a sequential flow.

        Args:
            agent_name (str): The name of the current agent.
            tasks (List[str]): The list of tasks in the flow.

        Returns:
            str: A string describing the agents ahead and behind in the sequence.
        """
        # Find the position of the current agent in the flow
        agent_position = None
        for i, task in enumerate(tasks):
            agent_names = [name.strip() for name in task.split(",")]
            if agent_name in agent_names:
                agent_position = i
                break

        if agent_position is None:
            return ""

        awareness_info = []

        # Check if there's an agent before (ahead in the sequence)
        if agent_position > 0:
            prev_task = tasks[agent_position - 1]
            prev_agents = [
                name.strip() for name in prev_task.split(",")
            ]
            if (
                prev_agents and prev_agents[0] != "H"
            ):  # Skip human agents
                awareness_info.append(
                    f"Agent ahead: {', '.join(prev_agents)}"
                )

        # Check if there's an agent after (behind in the sequence)
        if agent_position < len(tasks) - 1:
            next_task = tasks[agent_position + 1]
            next_agents = [
                name.strip() for name in next_task.split(",")
            ]
            if (
                next_agents and next_agents[0] != "H"
            ):  # Skip human agents
                awareness_info.append(
                    f"Agent behind: {', '.join(next_agents)}"
                )

        if awareness_info:
            return (
                f"Sequential awareness: {' | '.join(awareness_info)}"
            )
        return ""

    def _get_sequential_flow_info(self) -> str:
        """
        Gets information about the overall sequential flow structure.

        Returns:
            str: A string describing the sequential flow structure.
        """
        if not self.flow or "->" not in self.flow:
            return ""

        tasks = self.flow.split("->")
        flow_info = []

        for i, task in enumerate(tasks):
            agent_names = [name.strip() for name in task.split(",")]
            if (
                agent_names and agent_names[0] != "H"
            ):  # Skip human agents
                position_info = (
                    f"Step {i+1}: {', '.join(agent_names)}"
                )
                if i > 0:
                    prev_task = tasks[i - 1]
                    prev_agents = [
                        name.strip() for name in prev_task.split(",")
                    ]
                    if prev_agents and prev_agents[0] != "H":
                        position_info += (
                            f" (follows: {', '.join(prev_agents)})"
                        )
                if i < len(tasks) - 1:
                    next_task = tasks[i + 1]
                    next_agents = [
                        name.strip() for name in next_task.split(",")
                    ]
                    if next_agents and next_agents[0] != "H":
                        position_info += (
                            f" (leads to: {', '.join(next_agents)})"
                        )
                flow_info.append(position_info)

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

        tasks = self.flow.split("->")
        return self._get_sequential_awareness(agent_name, tasks)

    def get_sequential_flow_structure(self) -> str:
        """
        Gets the overall sequential flow structure information.

        Returns:
            str: A string describing the complete sequential flow structure.
        """
        return self._get_sequential_flow_info()

    def _run_concurrent_workflow(
        self,
        agent_names: List[str],
        img: str = None,
        streaming_callback: Optional[Callable[[str], None]] = None,
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

        Note:
            This method uses the run_agents_concurrently utility function
            to handle the actual parallel execution and result collection.
        """
        logger.info(f"Running agents in parallel: {agent_names}")

        # Get agent objects for concurrent execution
        agents_to_run = [
            self.agents[agent_name] for agent_name in agent_names
        ]

        # Prepare streaming wrappers per-agent if callback provided
        if streaming_callback is not None:
            # wrapper produces mini-json chunks following OpenAI style
            def make_agent_callback(name: str):
                def _on_chunk(chunk: str, done: bool = False):
                    try:
                        if chunk is None:
                            return
                        payload = {"agent": name, "chunk": chunk}
                        # send a single json chunk
                        streaming_callback(json.dumps(payload))
                        if done:
                            # send done marker for this agent
                            streaming_callback(json.dumps({"agent": name, "done": True}))
                    except Exception:
                        if self.verbose:
                            logger.exception("streaming callback failed")

                return _on_chunk

            # attach per-agent callbacks
            for a in agents_to_run:
                a.streaming_on = True
                a.streaming_callback = make_agent_callback(a.agent_name)

        # Run agents concurrently
        results = run_agents_concurrently(
            agents=agents_to_run,
            task=self.conversation.get_str(),
        )

        # Process results and update conversation
        response_dict = {}
        for i, agent_name in enumerate(agent_names):
            result = results[i]

            # print(f"Result: {result}")

            self.conversation.add(agent_name, result)
            response_dict[agent_name] = result
            logger.debug(f"Agent {agent_name} output: {result}")

        return response_dict

    def _run_sequential_workflow(
        self,
        agent_name: str,
        tasks: List[str],
        img: str = None,
        streaming_callback: Optional[Callable[[str], None]] = None,
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
            img (str, optional): Image input for agents that support it.
                Defaults to None.
            *args: Additional positional arguments passed to agent execution.
            **kwargs: Additional keyword arguments passed to agent execution.

        Returns:
            str: The result from the agent's execution, converted to string format.

        Note:
            If team_awareness is enabled, this method will add sequential awareness
            information to the conversation before executing the agent, informing
            the agent about its position in the workflow sequence.
        """
        logger.info(f"Running agent sequentially: {agent_name}")

        agent = self.agents[agent_name]

        # Add sequential awareness information for the agent
        awareness_info = self._get_sequential_awareness(
            agent_name, tasks
        )
        if awareness_info:
            self.conversation.add("system", awareness_info)
            logger.info(
                f"Added sequential awareness for {agent_name}: {awareness_info}"
            )

        # If a streaming callback is provided, set agent to streaming mode
        if streaming_callback is not None:
            agent.streaming_on = True

            def _agent_stream_wrapper(chunk: str, done: bool = False):
                try:
                    if chunk is None:
                        return
                    payload = {"agent": agent.agent_name, "chunk": chunk}
                    streaming_callback(json.dumps(payload))
                    if done:
                        streaming_callback(json.dumps({"agent": agent.agent_name, "done": True}))
                except Exception:
                    if self.verbose:
                        logger.exception("streaming callback failed")

            current_task = agent.run(
                task=self.conversation.get_str(),
                img=img,
                streaming_callback=_agent_stream_wrapper,
                *args,
                **kwargs,
            )
        else:
            current_task = agent.run(
                task=self.conversation.get_str(),
                img=img,
                *args,
                **kwargs,
            )
        current_task = any_to_str(current_task)

        self.conversation.add(agent.agent_name, current_task)

        return current_task

    def _run(
        self,
        task: str = None,
        img: str = None,
        streaming_callback: Optional[Callable[[str], None]] = None,
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

        Raises:
            ValueError: If flow validation fails or configuration is invalid.
            Exception: For any other errors during execution.

        Note:
            This method handles both sequential and concurrent execution patterns
            based on the flow syntax. It also supports custom task injection
            and multiple execution loops as configured.
        """
        try:
            self.conversation.add("User", task)

            if not self.validate_flow():
                logger.error("Flow validation failed")
                return "Invalid flow configuration."

            tasks = self.flow.split("->")
            response_dict = {}

            logger.info(
                f"Starting task execution with {len(tasks)} steps"
            )

            # Handle custom tasks
            if custom_tasks is not None:
                logger.info("Processing custom tasks")
                c_agent_name, c_task = next(
                    iter(custom_tasks.items())
                )
                position = tasks.index(c_agent_name)

                if position > 0:
                    tasks[position - 1] += "->" + c_task
                else:
                    tasks.insert(position, c_task)

            loop_count = 0
            while loop_count < self.max_loops:
                logger.info(
                    f"Starting loop {loop_count + 1}/{self.max_loops}"
                )

                for task_idx, task in enumerate(tasks):
                    agent_names = [
                        name.strip() for name in task.split(",")
                    ]

                    if len(agent_names) > 1:
                        # Concurrent processing - comma detected
                        concurrent_results = (
                            self._run_concurrent_workflow(
                                agent_names=agent_names,
                                img=img,
                                streaming_callback=streaming_callback,
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
                            tasks=tasks,
                            img=img,
                            streaming_callback=streaming_callback,
                            *args,
                            **kwargs,
                        )
                        response_dict[agent_name] = result

                loop_count += 1

            logger.info("Task execution completed")

            return history_output_formatter(
                conversation=self.conversation,
                type=self.output_type,
            )

        except Exception as e:
            self._catch_error(e)

    def _catch_error(self, e: Exception):
        """
        Handles errors that occur during swarm execution.

        Provides comprehensive error handling including logging, data persistence,
        and error reporting. This method is called whenever an exception occurs
        during the execution of the swarm.

        Args:
            e (Exception): The exception that occurred during execution.

        Returns:
            Exception: The original exception for potential re-raising.

        Note:
            If autosave is enabled, the current state of the swarm will be
            automatically saved to the logging system before error reporting.
        """
        if self.autosave is True:
            log_agent_data(self.to_dict())

        logger.error(
            f"AgentRearrange: Id: {self.id}, Name: {self.name}. An error occurred with your agent '{self.name}': Error: {e}. Traceback: {e.__traceback__}"
        )

        return e

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

        Note:
            This method automatically logs agent data before and after execution
            for telemetry and debugging purposes. Any exceptions are caught and
            handled by the _catch_error method.
        """
        try:
            log_agent_data(self.to_dict())

            out = self._run(
                task=task,
                img=img,
                *args,
                **kwargs,
            )

            log_agent_data(self.to_dict())

            return out

        except Exception as e:
            self._catch_error(e)

    def __call__(self, task: str, *args, **kwargs):
        """
        Make the class callable by executing the run() method.

        Enables the AgentRearrange instance to be called directly as a function,
        providing a convenient interface for task execution.

        Args:
            task (str): The task to execute through the agent workflow.
            *args: Additional positional arguments passed to run().
            **kwargs: Additional keyword arguments passed to run().

        Returns:
            The result from executing the task through the agent rearrange system.

        Example:
            >>> rearrange_system = AgentRearrange(agents=[agent1, agent2], flow="agent1 -> agent2")
            >>> result = rearrange_system("Process this data")
        """
        try:
            return self.run(task=task, *args, **kwargs)
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            return e

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

        Note:
            This method processes tasks in batches to manage resource usage.
            Each batch is processed sequentially, but individual tasks within
            a batch may run concurrently depending on the flow configuration.
        """
        try:
            results = []
            for i in range(0, len(tasks), batch_size):
                batch_tasks = tasks[i : i + batch_size]
                batch_imgs = (
                    img[i : i + batch_size]
                    if img
                    else [None] * len(batch_tasks)
                )

                # Process batch using concurrent execution
                batch_results = [
                    self.run(
                        task=task,
                        img=img_path,
                        *args,
                        **kwargs,
                    )
                    for task, img_path in zip(batch_tasks, batch_imgs)
                ]
                results.extend(batch_results)

            return results
        except Exception as e:
            self._catch_error(e)

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

        Note:
            This method uses ThreadPoolExecutor for true parallel execution.
            The number of concurrent executions is limited by max_workers parameter.
            Each task runs independently through the full agent workflow.
        """
        try:
            with ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                imgs = img if img else [None] * len(tasks)
                futures = [
                    executor.submit(
                        self.run,
                        task=task,
                        img=img_path,
                        *args,
                        **kwargs,
                    )
                    for task, img_path in zip(tasks, imgs)
                ]
                return [future.result() for future in futures]
        except Exception as e:
            self._catch_error(e)

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

        Raises:
            Exception: If an error occurs during task execution.

        Note:
            This method uses asyncio.to_thread to run the synchronous run method
            asynchronously, allowing integration with async/await patterns.
        """

        try:
            return await asyncio.to_thread(
                self.run, task=task, img=img, *args, **kwargs
            )
        except Exception as e:
            self._catch_error(e)

    def _serialize_callable(
        self, attr_value: Callable
    ) -> Dict[str, Any]:
        """
        Serializes callable attributes by extracting their name and docstring.

        This helper method handles the serialization of callable objects (functions,
        methods, etc.) by extracting their metadata for storage or logging purposes.

        Args:
            attr_value (Callable): The callable object to serialize.

        Returns:
            Dict[str, Any]: Dictionary containing the callable's name and docstring.
                Keys are "name" and "doc", values are the corresponding attributes.

        Note:
            This method is used internally by to_dict() to handle non-serializable
            callable attributes in a graceful manner.
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

        This helper method attempts to serialize individual attributes for storage
        or logging. It handles different types of objects including callables,
        objects with to_dict methods, and basic serializable types.

        Args:
            attr_name (str): The name of the attribute being serialized.
            attr_value (Any): The value of the attribute to serialize.

        Returns:
            Any: The serialized value of the attribute. For non-serializable objects,
                returns a string representation indicating the object type.

        Note:
            This method is used internally by to_dict() to handle various types
            of attributes in a robust manner, ensuring the serialization process
            doesn't fail on complex objects.
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

        This method provides a comprehensive serialization of the AgentRearrange
        instance, converting all attributes into a dictionary format suitable for
        storage, logging, or transmission. It handles complex objects gracefully
        by using helper methods for serialization.

        Returns:
            Dict[str, Any]: A dictionary representation of all class attributes.
                Non-serializable objects are converted to string representations
                or serialized using their to_dict method if available.

        Note:
            This method is used for telemetry logging and state persistence.
            It recursively handles nested objects and provides fallback handling
            for objects that cannot be directly serialized.
        """
        return {
            attr_name: self._serialize_attr(attr_name, attr_value)
            for attr_name, attr_value in self.__dict__.items()
        }


def rearrange(
    name: str = None,
    description: str = None,
    agents: List[Agent] = None,
    flow: str = None,
    task: str = None,
    img: str = None,
    *args,
    **kwargs,
):
    """
    Convenience function to create and execute an AgentRearrange system in one call.

    This function provides a simplified interface for creating an AgentRearrange
    instance and immediately executing a task with it. It's useful for quick
    prototyping or when you don't need to reuse the rearrange system.

    Parameters:
        name (str, optional): Name for the agent rearrange system.
            Defaults to None (uses AgentRearrange default).
        description (str, optional): Description of the system.
            Defaults to None (uses AgentRearrange default).
        agents (List[Agent]): The list of agents to be included in the system.
        flow (str): The flow pattern defining agent execution order.
            Uses '->' for sequential and ',' for concurrent execution.
        task (str, optional): The task to be performed during rearrangement.
            Defaults to None.
        img (str, optional): Image input for agents that support it.
            Defaults to None.
        *args: Additional positional arguments passed to AgentRearrange constructor.
        **kwargs: Additional keyword arguments passed to AgentRearrange constructor.

    Returns:
        The result of running the agent system with the specified task.
        The format depends on the output_type configuration.

    Example:
        >>> from swarms import Agent, rearrange
        >>>
        >>> # Create agents
        >>> agent1 = Agent(name="researcher", ...)
        >>> agent2 = Agent(name="writer", ...)
        >>> agent3 = Agent(name="reviewer", ...)
        >>>
        >>> # Execute task with flow
        >>> result = rearrange(
        ...     agents=[agent1, agent2, agent3],
        ...     flow="researcher -> writer, reviewer",
        ...     task="Research and write a report"
        ... )
    """
    agent_system = AgentRearrange(
        name=name,
        description=description,
        agents=agents,
        flow=flow,
        *args,
        **kwargs,
    )
    return agent_system.run(task=task, img=img)
