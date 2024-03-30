import logging
from collections import defaultdict
from swarms.utils.loguru_logger import logger
from swarms.structs.agent import Agent
from typing import Sequence, Callable, List, Dict, Union


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
        Initialize with a dictionary of Agent objects keyed by their names.
        """
        if not all(isinstance(agent, Agent) for agent in agents):
            raise ValueError(
                "All elements must be instances of the Agent class."
            )
        self.agents = agents
        self.verbose = verbose
        self.custom_prompt = custom_prompt
        self.callbacks = callbacks
        self.flows = defaultdict(list)

    def parse_pattern(self, pattern: str):
        """
        Parse the interaction pattern and setup task flows.

        Pattern format: "a -> b, c -> d, e -> f"
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
                    logging.error(
                        f"Source agent {source_name} not found."
                    )
                    return False

                destinations_names = destinations_str.split()
                for dest_name in destinations_names:
                    dest = self.find_agent_by_name(dest_name)
                    if dest is None:
                        logging.error(
                            f"Destination agent {dest_name} not"
                            " found."
                        )
                        return False
                    self.flows[source.agent_name].append(
                        dest.agent_name
                    )
            return True
        except Exception as e:
            logger.error(f"Error: {e}")
            raise e

    def self_find_agen_by_name(self, name: str):
        for agent in self.agents:
            if agent.agent_name == name:
                return agent
        return None

    def __call__(
        self,
        agents: Sequence[Agent] = None,
        pattern: str = None,
        task: str = None,
        *args,
        **kwargs,
    ):
        """
        Execute the task based on the specified pattern.
        """
        try:
            if agents:
                self.flows.clear()  # Reset previous flows
                if not self.parse_pattern(pattern):
                    return  # Pattern parsing failed

                for source, destinations in self.flows.items():
                    for dest in destinations:
                        # agents[dest].runt(f"{task} (from {source})")
                        dest_agent = self.self_find_agen_by_name(dest)

                        if self.custom_prompt:
                            dest_agent.run(
                                f"{task} {self.custom_prompt}"
                            )
                        else:
                            dest_agent.run(f"{task} (from {source})")

            else:
                self.flows.clear()  # Reset previous flows
                if not self.parse_pattern(pattern):
                    return  # Pattern parsing failed

                for source, destinations in self.flows.items():
                    for dest in destinations:
                        dest_agent = self.self_find_agen_by_name(dest)
                        if self.custom_prompt:
                            dest_agent.run(
                                f"{task} {self.custom_prompt}"
                            )
                        else:
                            dest_agent.run(f"{task} (from {source})")
        except Exception as e:
            logger.error(
                f"Error: {e} try again by providing agents and"
                " pattern"
            )
            raise e


# # Example usage
# try:
#     agents = [
#         Agent(name=f"b{i}") for i in range(1, 4)
#     ]  # Creating agents b1, b2, b3
#     agents.append(Agent(name="d"))  # Adding agent d
#     rearranger = Rearrange(agents)

#     # Specifying a complex pattern for task execution
#     rearranger.execute("d -> b1 b2 b3, b2 -> b3", "Analyze data")
# except ValueError as e:
#     logging.error(e)
