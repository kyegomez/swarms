from typing import Callable, Dict, List, Optional

from swarms.memory.base_vectordb import BaseVectorDatabase
from swarms.structs.agent import Agent
from swarms.structs.base_swarm import BaseSwarm
from swarms.utils.loguru_logger import logger


class AgentRearrange(BaseSwarm):
    """
    A class representing a swarm of agents for rearranging tasks.

    Attributes:
        agents (dict): A dictionary of agents, where the key is the agent's name and the value is the agent object.
        flow (str): The flow pattern of the tasks.

    Methods:
        __init__(agents: List[Agent] = None, flow: str = None): Initializes the AgentRearrange object.
        add_agent(agent: Agent): Adds an agent to the swarm.
        remove_agent(agent_name: str): Removes an agent from the swarm.
        add_agents(agents: List[Agent]): Adds multiple agents to the swarm.
        validate_flow(): Validates the flow pattern.
        run(task): Runs the swarm to rearrange the tasks.
    """

    def __init__(
        self,
        agents: List[Agent] = None,
        flow: str = None,
        max_loops: int = 1,
        verbose: bool = True,
        memory_system: BaseVectorDatabase = None,
        human_in_the_loop: bool = False,
        custom_human_in_the_loop: Optional[Callable[[str], str]] = None,
        *args,
        **kwargs,
    ):
        """
        Initializes the AgentRearrange object.

        Args:
            agents (List[Agent], optional): A list of Agent objects. Defaults to None.
            flow (str, optional): The flow pattern of the tasks. Defaults to None.
        """
        self.agents = {agent.name: agent for agent in agents}
        self.flow = flow if flow is not None else ""
        self.verbose = verbose
        self.max_loops = max_loops if max_loops > 0 else 1
        self.memory_system = memory_system
        self.human_in_the_loop = human_in_the_loop
        self.custom_human_in_the_loop = custom_human_in_the_loop
        self.swarm_history = {agent.agent_name: [] for agent in agents}

        # Verbose is True
        if verbose is True:
            logger.add("agent_rearrange.log")

        # Memory system
        if memory_system is not None:
            for agent in self.agents.values():
                agent.long_term_memory = memory_system

        logger.info(
            "AgentRearrange initialized with agents: {}".format(
                list(self.agents.keys())
            )
        )

    def add_agent(self, agent: Agent):
        """
        Adds an agent to the swarm.

        Args:
            agent (Agent): The agent to be added.
        """
        logger.info(f"Adding agent {agent.name} to the swarm.")
        self.agents[agent.name] = agent

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
            self.agents[agent.name] = agent

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
                if agent_name not in self.agents and agent_name != "H":
                    raise ValueError(
                        f"Agent '{agent_name}' is not registered."
                    )
                agents_in_flow.append(agent_name)

        # If the length of the agents does not equal the length of the agents in flow
        if len(set(agents_in_flow)) != len(agents_in_flow):
            raise ValueError(
                "Duplicate agent names in the flow are not allowed."
            )

        print("Flow is valid.")
        return True

    def run(
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
            task: The initial task to be processed.

        Returns:
            str: The final processed task.
        """
        try:
            if not self.validate_flow():
                return "Invalid flow configuration."

            tasks = self.flow.split("->")
            current_task = task

            # If custom_tasks have the agents name and tasks then combine them
            if custom_tasks is not None:
                c_agent_name, c_task = next(iter(custom_tasks.items()))

                # Find the position of the custom agent in the tasks list
                position = tasks.index(c_agent_name)

                # If there is a prebois agent merge its task with the custom tasks
                if position > 0:
                    tasks[position - 1] += "->" + c_task
                else:
                    # If there is no prevous agent just insert the custom tasks
                    tasks.insert(position, c_task)

            # Set the loop counter
            loop_count = 0
            while loop_count < self.max_loops:
                for task in tasks:
                    agent_names = [
                        name.strip() for name in task.split(",")
                    ]
                    if len(agent_names) > 1:
                        # Parallel processing
                        logger.info(
                            f"Running agents in parallel: {agent_names}"
                        )
                        results = []
                        for agent_name in agent_names:
                            if agent_name == "H":
                                # Human in the loop intervention
                                if (
                                    self.human_in_the_loop
                                    and self.custom_human_in_the_loop
                                ):
                                    current_task = (
                                        self.custom_human_in_the_loop(
                                            current_task
                                        )
                                    )
                                else:
                                    current_task = input(
                                        "Enter your response:"
                                    )
                            else:
                                agent = self.agents[agent_name]
                                result = agent.run(
                                    current_task, img, *args, **kwargs
                                )
                                results.append(result)

                        current_task = "; ".join(results)
                    else:
                        # Sequential processing
                        logger.info(
                            f"Running agents sequentially: {agent_names}"
                        )
                        agent_name = agent_names[0]
                        if agent_name == "H":
                            # Human-in-the-loop intervention
                            if (
                                self.human_in_the_loop
                                and self.custom_human_in_the_loop
                            ):
                                current_task = (
                                    self.custom_human_in_the_loop(
                                        current_task
                                    )
                                )
                            else:
                                current_task = input(
                                    "Enter the next task: "
                                )
                        else:
                            agent = self.agents[agent_name]
                            current_task = agent.run(
                                current_task, img, *args, **kwargs
                            )
                loop_count += 1

            return current_task
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            return e


def rearrange(
    agents: List[Agent] = None,
    flow: str = None,
    task: str = None,
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
        agents=agents, flow=flow, *args, **kwargs
    )
    return agent_system.run(task, *args, **kwargs)


# out = AgentRearrange(
#     agents=[agent1, agent2, agent3],
#     flow="agent1 -> agent2, agent3, swarm",
#     task="Perform a task",
#     swarm = "agent1 -> agent2, agent3, swarm"

# )
