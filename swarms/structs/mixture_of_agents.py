from swarms.structs.agent import Agent
from swarms.structs.base_swarm import BaseSwarm
from typing import List, Any

from swarms.structs.conversation import Conversation
from pydantic import BaseModel
from swarms.utils.loguru_logger import logger


class AgentRun(BaseModel):
    agent_name: str
    output: Any


class Metadata(BaseModel):
    layers: int
    agent_runs: List[AgentRun]
    final_output: Any


class MixtureOfAgents(BaseSwarm):
    """
    Represents a mixture of agents in a swarm.
    The process is parallel -> sequential -> parallel -> final output agent.
    From the paper: https://arxiv.org/pdf/2406.04692

    Attributes:
        agents (List[Agent]): The list of agents in the swarm.
        flow (str): The flow of the swarm.
        max_loops (int): The maximum number of loops to run.
        verbose (bool): Flag indicating whether to print verbose output.
        layers (int, optional): The number of layers in the swarm. Defaults to None.
        rules (str, optional): The rules for the swarm. Defaults to None.
    """

    def __init__(
        self,
        name: str = "MixtureOfAgents",
        description: str = "A swarm of agents that run in parallel and sequentially.",
        agents: List[Agent] = None,
        max_loops: int = 1,
        verbose: bool = True,
        layers: int = 3,
        rules: str = None,
        final_agent: Agent = None,
        auto_save: bool = False,
        saved_file_name: str = "moe_swarm.json",
    ):
        self.name = name
        self.description = description
        self.agents = agents
        self.max_loops = max_loops
        self.verbose = verbose
        self.layers = layers
        self.rules = rules
        self.final_agent = final_agent
        self.auto_save = auto_save
        self.saved_file_name = saved_file_name

        # Check the agents
        self.agent_check()
        self.final_agent_check()

        # Conversation
        self.conversation = Conversation(
            time_enabled=True,
            rules=rules,
        )

        # Initialize the swarm
        self.swarm_initialization()

    def agent_check(self):
        try:
            if not isinstance(self.agents, list):
                raise TypeError("Input must be a list of agents.")
            for agent in self.agents:
                if not isinstance(agent, Agent):
                    raise TypeError(
                        "Input must be a list of agents."
                        "Each agent must be an instance of Agent."
                    )
        except TypeError as e:
            logger.error(f"Error checking agents: {e}")

    def final_agent_check(self):
        try:
            if not isinstance(self.final_agent, Agent):
                raise TypeError(
                    "Final agent must be an instance of Agent."
                )
        except TypeError as e:
            logger.error(f"Error checking final agent: {e}")

    def swarm_initialization(self):
        # Name, description, and logger
        logger.info(f"Initializing swarm {self.name}.")
        logger.info(f"Description: {self.description}")
        logger.info(f"Initializing swarm with {len(self.agents)} agents.")

    def run(self, task: str = None, *args, **kwargs):
        try:
            # Running the swarm
            logger.info(f"Running swarm {self.name}.")

            self.conversation.add("user", task)

            # Conversation history
            history = self.conversation.return_history_as_string()

            agent_runs = []
            layer = 0
            while layer < self.layers:
                logger.info(f"Running layer {layer} of the swarm.")
                # Different Layers
                # Run the agents for all agents on the input
                responses = []
                for agent in self.agents:
                    out = agent.run(history, *args, **kwargs)
                    responses.append((agent.agent_name, out))
                    agent_runs.append(
                        AgentRun(agent_name=agent.agent_name, output=out)
                    )

                    # Log the agent run
                    logger.info(f"Agent {agent.agent_name} output: {out}")

                # Add all the responses to the conversation
                logger.info("Adding responses to the conversation.")
                for agent_name, response in responses:
                    self.conversation.add(agent_name, response)

                # Update the history
                history = self.conversation.return_history_as_string()

                layer += 1

                logger.info(f"Completed layer {layer} of the swarm.")

            # Run the final output agent on the entire conversation history
            logger.info(
                "Running the final output agent on the conversation history."
            )
            final_output = self.final_agent.run(history, *args, **kwargs)
            self.conversation.add(
                self.final_agent.agent_name, final_output
            )

            # Create metadata
            logger.info("Creating metadata for the swarm.")
            metadata = Metadata(
                layers=self.layers,
                agent_runs=agent_runs,
                final_output=final_output,
            )

            # Save metadata to JSON file
            logger.info("Saving metadata to JSON file.")
            with open(self.saved_file_name, "w") as f:
                f.write(metadata.json())

            return self.conversation.return_history_as_string()
        except Exception as e:
            logger.error(
                f"Error running swarm: {e} try optimizing the swarm inputs or re-iterate on the task."
            )
        return None
