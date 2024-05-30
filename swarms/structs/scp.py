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
    def __init__(
        self,
        agents: List[AgentType],
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
    ):
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
    ):
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
        agent.receieve_mesage(self.message_log(agent, message))
