from swarms.structs.agent import Agent
from swarms.structs.base_swarm import BaseSwarm
from typing import List, Any

from swarms.structs.conversation import Conversation
from pydantic import BaseModel
from swarms.utils.loguru_logger import logger
from swarms.memory.base_vectordb import BaseVectorDatabase


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
        scp: BaseVectorDatabase = None,
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
        self.scp = scp

        # Check the agents
        self.reliability_check()
        self.agent_check()
        self.final_agent_check()

        # Conversation
        self.conversation = Conversation(
            time_enabled=True,
            rules=rules,
        )

        # Initialize the swarm
        self.swarm_initialization()

        # Communication Protocol
        self.communication_protocol()

    def reliability_check(self):
        if self.final_agent is None:
            raise ValueError("Final agent is not defined.")

        if self.agents is None:
            raise ValueError("Agents are not defined.")

        if self.layers is None:
            raise ValueError("Layers are not defined.")

    def communication_protocol(self):
        try:
            # Memory system
            logger.info(
                "Initializing SCP --- Swarm Communication Protocol"
            )

            if self.scp is not None:
                for agent in self.agents.values():
                    agent.long_term_memory = self.scp
                    logger.info("Agents have been integrated with SCP:")
        except Exception as e:
            logger.error(f"Error initializing SCP: {e}")

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
        """
        Initializes the swarm by logging the swarm name, description, and the number of agents.
        """
        # Name, description, and logger
        logger.info(f"Initializing Mixture of Agents Swarm: {self.name}.")
        logger.info(f"Description: {self.description}")
        logger.info(f"Initializing swarm with {len(self.agents)} agents.")

    def run(self, task: str = None, *args, **kwargs):
        """
        Runs the swarm with the given task and returns the conversation history.

        Args:
            task (str): The task to be performed by the swarm.

        Returns:
            str: The conversation history as a string.
        """
        try:
            # Running the swarm
            logger.info(f"Running swarm {self.name}.")

            self.conversation.add("user", task)
            # self.scp.add(f"User: {task}")

            # Conversation history
            history = self.conversation.return_history_as_string()
            # self.scp.add(f"Conversation History: {history}")

            agent_runs = []
            layer = 0
            while layer < self.layers:
                logger.info(f"Running layer {layer} of the swarm.")
                # Different Layers
                # Run the agents for all agents on the input
                responses = []
                for agent in self.agents:
                    out = agent.run(history, *args, **kwargs)
                    # self.scp.add(
                    #     f"Agent: {agent.agent_name} Output: {out}"
                    # )
                    responses.append((agent.agent_name, out))
                    agent_runs.append(
                        AgentRun(agent_name=agent.agent_name, output=out)
                    )

                    # Log the agent run
                    # logger.info(f"Agent {agent.agent_name} output: {out}")

                # Add all the responses to the conversation
                logger.info("Adding responses to the conversation.")
                for agent_name, response in responses:
                    self.conversation.add(agent_name, response)

                # Update the history
                history = self.conversation.return_history_as_string()
                # self.scp.add(f"Conversation History: {history}")

                layer += 1

                logger.info(f"Completed layer {layer} of the swarm.")

            # Run the final output agent on the entire conversation history
            logger.info(
                "Running the final output agent on the conversation history."
            )
            final_output = self.final_agent.run(history, *args, **kwargs)
            # self.scp.add(
            #     f"Final Agent: {self.final_agent.agent_name} Output: {final_output}"
            # )
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
