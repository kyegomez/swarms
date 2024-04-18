import logging
from collections import defaultdict
from swarms.utils.loguru_logger import logger
from swarms.structs.agent import Agent
from typing import Sequence, Callable


class AgentRearrange:
    def __init__(
        self,
        agents: Sequence[Agent] = None,
        verbose: bool = False,
        custom_prompt: str = None,
        callbacks: Sequence[Callable] = None,
        *args,
        **kwargs,
    ):
        """
        Initialize the AgentRearrange class.

        Args:
            agents (Sequence[Agent], optional): A sequence of Agent objects. Defaults to None.
            verbose (bool, optional): Whether to enable verbose mode. Defaults to False.
            custom_prompt (str, optional): A custom prompt string. Defaults to None.
            callbacks (Sequence[Callable], optional): A sequence of callback functions. Defaults to None.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        if not all(isinstance(agent, Agent) for agent in agents):
            raise ValueError(
                "All elements must be instances of the Agent class."
            )
        self.agents = agents
        self.verbose = verbose
        self.custom_prompt = custom_prompt
        self.callbacks = callbacks if callbacks is not None else []
        self.flows = defaultdict(list)

    def parse_pattern(self, pattern: str):
        """
        Parse the interaction pattern and setup task flows.

        Args:
            pattern (str): The interaction pattern to parse.

        Returns:
            bool: True if the pattern parsing is successful, False otherwise.
        """
        try:
            for flow in pattern.split(","):
                parts = [part.strip() for part in flow.split("->")]
                if len(parts) != 2:
                    logging.error(
                        f"Invalid flow pattern: {flow}. Each flow"
                        " must have exactly one '->'."
                    )
                    return False

                source_name, destinations_str = parts
                source = self.find_agent_by_name(source_name)
                if source is None:
                    logging.error(f"Source agent {source_name} not found.")
                    return False

                destinations_names = destinations_str.split()
                for dest_name in destinations_names:
                    dest = self.find_agent_by_name(dest_name)
                    if dest is None:
                        logging.error(
                            f"Destination agent {dest_name} not" " found."
                        )
                        return False
                    self.flows[source.agent_name].append(dest.agent_name)
            return True
        except Exception as e:
            logger.error(f"Error: {e}")
            raise e

    def self_find_agen_by_name(self, name: str):
        """
        Find an agent by its name.

        Args:
            name (str): The name of the agent to find.

        Returns:
            Agent: The Agent object if found, None otherwise.
        """
        for agent in self.agents:
            if agent.agent_name == name:
                return agent
        return None

    def __call__(
        self,
        agents: Sequence[Agent] = None,
        pattern: str = None,
        task: str = None,
        **tasks,
    ):
        """
        Execute the task based on the specified pattern.

        Args:
            agents (Sequence[Agent], optional): A sequence of Agent objects. Defaults to None.
            pattern (str, optional): The interaction pattern to follow. Defaults to None.
            task (str, optional): The task to execute. Defaults to None.
            **tasks: Additional tasks specified as keyword arguments.
        """
        try:
            if agents:
                self.flows.clear()  # Reset previous flows
                if not self.parse_pattern(pattern):
                    return  # Pattern parsing failed

                for source, destinations in self.flows.items():
                    for dest in destinations:
                        dest_agent = self.self_find_agen_by_name(dest)
                        task = tasks.get(dest, task)

                        if self.custom_prompt:
                            dest_agent.run(f"{task} {self.custom_prompt}")
                        else:
                            dest_agent.run(f"{task} (from {source})")
            # else:
            #     raise ValueError(
            #         "No agents provided. Please provide agents to"
            #         " execute the task."
            #     )
        except Exception as e:
            logger.error(
                f"Error: {e} try again by providing agents and" " pattern"
            )
            raise e


# # Example usage
# try:
#     agents = [
#         Agent(agent_name=f"b{i}") for i in range(1, 4)
#     ]  # Creating agents b1, b2, b3
#     agents.append(Agent(agent_name="d"))  # Adding agent d
#     rearranger = Rearrange(agents)

#     # Specifying a complex pattern for task execution
#     rearranger.execute("d -> b1 b2 b3, b2 -> b3", "Analyze data")
# except ValueError as e:
#     logging.error(e)
