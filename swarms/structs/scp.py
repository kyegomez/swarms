"""
Swarm Communication Protocol
- RAG as a communication protocol
- Each Agent is connected to a database so they can see each others 
memories, actions, and experiences

"""

import json
from swarms.structs.omni_agent_types import AgentType
from swarms.structs.base_structure import BaseStructure
from typing import List
from swarms.memory.base_vectordb import BaseVectorDatabase
import time
from swarms.utils.loguru_logger import logger
from pydantic import BaseModel, Field
from typing import Any


class SwarmCommunicationProtocol(BaseModel):
    agent_name: str = Field(
        None, title="Agent Name", description="The name of the agent"
    )
    message: str = Field(
        None, title="Message", description="The message to be sent"
    )
    timestamp: float = Field(
        None,
        title="Timestamp",
        description="The time the message was sent",
    )


class SCP(BaseStructure):
    """
    Represents the Swarm Communication Protocol (SCP).

    SCP is responsible for managing agents and their communication within a swarm.

    Args:
        agents (List[AgentType]): A list of agents participating in the swarm.
        memory_system (BaseVectorDatabase, optional): The memory system used by the agents. Defaults to None.

    Attributes:
        agents (List[AgentType]): A list of agents participating in the swarm.
        memory_system (BaseVectorDatabase): The memory system used by the agents.

    Methods:
        message_log(agent: AgentType, task: str = None, message: str = None) -> str:
            Logs a message from an agent and adds it to the memory system.

        run_single_agent(agent: AgentType, task: str, *args, **kwargs) -> Any:
            Runs a task for a single agent and logs the output.

        send_message(agent: AgentType, message: str):
            Sends a message to an agent and logs it.

    """

    def __init__(
        self,
        agents: List[AgentType] = None,
        memory_system: BaseVectorDatabase = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.agents = agents
        self.memory_system = memory_system

        # Memory system
        if memory_system is not None:
            for agent in self.agents.values():
                agent.long_term_memory = memory_system

        logger.info(
            "AgentRearrange initialized with agents: {}".format(
                list(self.agents.keys())
            )
        )

    def message_log(
        self, agent: AgentType, task: str = None, message: str = None
    ) -> str:
        """
        Logs a message from an agent and adds it to the memory system.

        Args:
            agent (AgentType): The agent that generated the message.
            task (str, optional): The task associated with the message. Defaults to None.
            message (str, optional): The message content. Defaults to None.

        Returns:
            str: The JSON-encoded log message.

        """
        log = {
            "agent_name": agent.agent_name,
            "task": task,
            "response": message,
            "timestamp": time.time(),
        }

        # Transform the log into a string
        log_output = json.dumps(log)

        # Add the log to the memory system
        self.memory_system.add(log)

        return log_output

    def run_single_agent(
        self, agent: AgentType, task: str, *args, **kwargs
    ) -> Any:
        """
        Runs a task for a single agent and logs the output.

        Args:
            agent (AgentType): The agent to run the task for.
            task (str): The task to be executed.

        Returns:
            Any: The output of the task.

        """
        # Send the message to the agent
        output = agent.run(task)

        # log the message
        self.message_log(
            agent=agent,
            task=task,
            message=output,
        )

        # Log the output
        return output

    def send_message(self, agent: AgentType, message: str):
        """
        Sends a message to an agent and logs it.

        Args:
            agent (AgentType): The agent to send the message to.
            message (str): The message to be sent.

        """
        agent.receieve_mesage(self.message_log(agent, message))
