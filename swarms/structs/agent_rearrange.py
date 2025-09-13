import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Union

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.telemetry.main import log_agent_data
from swarms.utils.any_to_str import any_to_str
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.loguru_logger import initialize_logger
from swarms.utils.output_types import OutputType
from swarms.structs.swarm_id import swarm_id

logger = initialize_logger(log_folder="rearrange")


class AgentRearrange:

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
        streaming_callback: Optional[Callable[[str], None]] = None,
        *args,
        **kwargs,
    ):
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
        self.streaming_callback = streaming_callback

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
        if self.agents is None or len(self.agents) == 0:
            raise ValueError("Agents list cannot be None or empty")

        if self.max_loops == 0:
            raise ValueError("max_loops cannot be 0")

        if self.flow is None or self.flow == "":
            raise ValueError("flow cannot be None or empty")

        if self.output_type is None or self.output_type == "":
            raise ValueError("output_type cannot be None or empty")

    def set_custom_flow(self, flow: str):
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

    def _run(
        self,
        task: str = None,
        img: str = None,
        custom_tasks: Dict[str, str] = None,
        *args,
        **kwargs,
    ):
        """
        Runs the swarm to rearrange the tasks.

        Args:
            task (str, optional): The initial task to be processed. Defaults to None.
            img (str, optional): Image input for agents that support it. Defaults to None.
            custom_tasks (Dict[str, str], optional): Custom tasks for specific agents. Defaults to None.
            output_type (str, optional): Format of the output. Can be:
                - "all": String containing all agent responses concatenated
                - "final": Only the final agent's response
                - "list": List of all agent responses
                - "dict": Dict mapping agent names to their responses
                Defaults to "final".
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Union[str, List[str], Dict[str, str]]: The processed output in the specified format

        Raises:
            ValueError: If flow validation fails
            Exception: For any other errors during execution
        """
        try:
            self.conversation.add("User", task)

            if not self.validate_flow():
                logger.error("Flow validation failed")
                return "Invalid flow configuration."

            tasks = self.flow.split("->")
            current_task = task
            response_dict = {}

            logger.info(
                f"Starting task execution with {len(tasks)} steps"
            )

            # # Handle custom tasks
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
                        # Parallel processing
                        logger.info(
                            f"Running agents in parallel: {agent_names}"
                        )

                        for agent_name in agent_names:
                            agent = self.agents[agent_name]
                            # Set agent.streaming_on if no streaming_callback
                            if self.streaming_callback is not None:
                                agent.streaming_on = True
                            result = agent.run(
                                task=self.conversation.get_str(),
                                img=img,
                                *args,
                                **kwargs,
                            )
                            result = any_to_str(result)


                            # Call streaming callback with the result if provided
                            if self.streaming_callback:
                                self.streaming_callback(result)

                            self.conversation.add(
                                agent.agent_name, result
                            )

                            response_dict[agent_name] = result
                            logger.debug(
                                f"Agent {agent_name} output: {result}"
                            )

                        ",".join(agent_names)

                    else:
                        # Sequential processing
                        logger.info(
                            f"Running agent sequentially: {agent_names[0]}"
                        )
                        agent_name = agent_names[0]

                        agent = self.agents[agent_name]

                        # Add sequential awareness information for the agent
                        awareness_info = (
                            self._get_sequential_awareness(
                                agent_name, tasks
                            )
                        )
                        if awareness_info:
                            self.conversation.add(
                                "system", awareness_info
                            )
                            logger.info(
                                f"Added sequential awareness for {agent_name}: {awareness_info}"
                            )

                        # Set agent.streaming_on if no streaming_callback
                        if self.streaming_callback is not None:
                            agent.streaming_on = True
                        current_task = agent.run(
                            task=self.conversation.get_str(),
                            img=img,
                            *args,
                            **kwargs,
                        )
                        current_task = any_to_str(current_task)

                        # Call streaming callback with the result if provided
                        if self.streaming_callback:
                            self.streaming_callback(current_task)

                        self.conversation.add(
                            agent.agent_name, current_task
                        )

                        response_dict[agent_name] = current_task

                loop_count += 1

            logger.info("Task execution completed")

            return history_output_formatter(
                conversation=self.conversation,
                type=self.output_type,
            )

        except Exception as e:
            self._catch_error(e)

    def _catch_error(self, e: Exception):
        if self.autosave is True:
            log_agent_data(self.to_dict())

        logger.error(
            f"An error occurred with your swarm {self.name}: Error: {e} Traceback: {e.__traceback__}"
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
        Execute the agent rearrangement task with specified compute resources.

        Args:
            task (str, optional): The task to execute. Defaults to None.
            img (str, optional): Path to input image if required. Defaults to None.
            *args: Additional positional arguments passed to _run().
            **kwargs: Additional keyword arguments passed to _run().

        Returns:
            The result from executing the task through the cluster operations wrapper.
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

        Args:
            task (str): The task to execute.
            *args: Additional positional arguments passed to run().
            **kwargs: Additional keyword arguments passed to run().

        Returns:
            The result from executing run().
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
        Process multiple tasks in batches.

        Args:
            tasks: List of tasks to process
            img: Optional list of images corresponding to tasks
            batch_size: Number of tasks to process simultaneously
            device: Computing device to use
            device_id: Specific device ID if applicable
            all_cores: Whether to use all CPU cores
            all_gpus: Whether to use all available GPUs

        Returns:
            List of results corresponding to input tasks
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

        Args:
            tasks: List of tasks to process
            img: Optional list of images corresponding to tasks
            max_workers: Maximum number of worker threads
            device: Computing device to use
            device_id: Specific device ID if applicable
            all_cores: Whether to use all CPU cores
            all_gpus: Whether to use all available GPUs

        Returns:
            List of results corresponding to input tasks
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
    Rearranges the given list of agents based on the specified flow.

    Parameters:
        agents (List[Agent]): The list of agents to be rearranged.
        flow (str): The flow used for rearranging the agents.
        task (str, optional): The task to be performed during rearrangement. Defaults to None.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        The result of running the agent system with the specified task.

    Example:
        agents = [agent1, agent2, agent3]
        flow = "agent1 -> agent2, agent3"
        task = "Perform a task"
        rearrange(agents, flow, task)
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
