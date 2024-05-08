from swarms import Agent
from typing import List
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
    ):
        """
        Initializes the AgentRearrange object.

        Args:
            agents (List[Agent], optional): A list of Agent objects. Defaults to None.
            flow (str, optional): The flow pattern of the tasks. Defaults to None.
        """
        self.agents = {agent.name: agent for agent in agents}
        self.flow = flow
        self.verbose = verbose
        self.max_loops = max_loops

        if verbose is True:
            logger.add("agent_rearrange.log")

    def add_agent(self, agent: Agent):
        """
        Adds an agent to the swarm.

        Args:
            agent (Agent): The agent to be added.
        """
        logger.info(f"Adding agent {agent.name} to the swarm.")
        self.agents[agent.name] = agent

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
        tasks = self.flow.split("->")
        for task in tasks:
            agent_names = [name.strip() for name in task.split(",")]
            for agent_name in agent_names:
                if agent_name not in self.agents:
                    raise ValueError(
                        f"Agent '{agent_name}' is not registered."
                    )
                agents_in_flow.append(agent_name)

        if len(set(agents_in_flow)) != len(agents_in_flow):
            raise ValueError(
                "Duplicate agent names in the flow are not allowed."
            )

        print("Flow is valid.")
        return True

    def run(self, task: str, *args, **kwargs):
        """
        Runs the swarm to rearrange the tasks.

        Args:
            task: The initial task to be processed.

        Returns:
            str: The final processed task.
        """
        if not self.validate_flow():
            return "Invalid flow configuration."

        tasks = self.flow.split("->")
        current_task = task

        for task in tasks:
            agent_names = [name.strip() for name in task.split(",")]
            if len(agent_names) > 1:
                # Parallel processing
                logger.info(f"Running agents in parallel: {agent_names}")
                results = []
                for agent_name in agent_names:
                    agent = self.agents[agent_name]
                    result = agent.run(current_task, *args, **kwargs)
                    results.append(result)
                current_task = "; ".join(results)
            else:
                # Sequential processing
                logger.info(f"Running agents sequentially: {agent_names}")
                agent = self.agents[agent_names[0]]
                current_task = agent.run(current_task, *args, **kwargs)

        return current_task


def rearrange(
    agents: List[Agent], flow: str, task: str = None, *args, **kwargs
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


# # Initialize the director agent
# director = Agent(
#     agent_name="Director",
#     system_prompt="Directs the tasks for the workers",
#     llm=Anthropic(),
#     max_loops=1,
#     dashboard=False,
#     streaming_on=True,
#     verbose=True,
#     stopping_token="<DONE>",
#     state_save_file_type="json",
#     saved_state_path="director.json",
# )

# # Initialize worker 1
# worker1 = Agent(
#     agent_name="Worker1",
#     system_prompt="Generates a transcript for a youtube video on what swarms are",
#     llm=Anthropic(),
#     max_loops=1,
#     dashboard=False,
#     streaming_on=True,
#     verbose=True,
#     stopping_token="<DONE>",
#     state_save_file_type="json",
#     saved_state_path="worker1.json",
# )

# # Initialize worker 2
# worker2 = Agent(
#     agent_name="Worker2",
#     system_prompt="Summarizes the transcript generated by Worker1",
#     llm=Anthropic(),
#     max_loops=1,
#     dashboard=False,
#     streaming_on=True,
#     verbose=True,
#     stopping_token="<DONE>",
#     state_save_file_type="json",
#     saved_state_path="worker2.json",
# )


# flow = "Director -> Worker1 -> Worker2"
# agent_system = AgentRearrange(
#     agents=[director, worker1, worker2], flow=flow
# )
# # Run the system
# output = agent_system.run(
#     "Create a format to express and communicate swarms of llms in a structured manner for youtube"
# )
