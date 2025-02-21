import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError

from swarms import Agent
from swarms.utils.loguru_logger import logger

class AgentConfigSchema(BaseModel):
    uuid: str = Field(
        ..., description="The unique identifier for the agent."
    )
    name: str = None
    description: str = None
    time_added: str = Field(
        time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        description="Time when the agent was added to the registry."
    )
    config: Dict[Any, Any] = None

class AgentRegistrySchema(BaseModel):
    name: str
    description: str
    agents: List[AgentConfigSchema]
    time_registry_created: str = Field(
        time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        description="Time when the registry was created."
    )
    number_of_agents: int = Field(
        0, description="The number of agents in the registry."
    )

class AgentRegistry:
    def __init__(
        self,
        name: str = "Agent Registry",
        description: str = "A registry for managing agents.",
        agents: Optional[List[Agent]] = None,
        return_json: bool = True,
        auto_save: bool = False,
        *args,
        **kwargs,
    ):
        self.name = name
        self.description = description
        self.return_json = return_json
        self.auto_save = auto_save
        self.agents: Dict[str, Agent] = {}
        self.lock = Lock()
        
        self.agent_registry = AgentRegistrySchema(
            name=self.name,
            description=self.description,
            agents=[],
            number_of_agents=len(agents) if agents else 0,
        )
        
        if agents:
            self.add_many(agents)

    def return_all_agents(self) -> List[Agent]:
        try:
            with self.lock:
                agents = list(self.agents.values())
                logger.info("Returning all agents.")
                return agents
        except Exception as e:
            logger.error(f"Error: {e}")
            raise e
