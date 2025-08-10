import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError

from swarms.structs.agent import Agent
from swarms.utils.loguru_logger import logger


class AgentConfigSchema(BaseModel):
    uuid: str = Field(
        ...,
        description="The unique identifier for the agent.",
    )
    name: str = None
    description: str = None
    time_added: str = Field(
        time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        description="Time when the agent was added to the registry.",
    )
    config: Dict[Any, Any] = None


class AgentRegistrySchema(BaseModel):
    name: str
    description: str
    agents: List[AgentConfigSchema]
    time_registry_creatd: str = Field(
        time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        description="Time when the registry was created.",
    )
    number_of_agents: int = Field(
        0,
        description="The number of agents in the registry.",
    )


class AgentRegistry:
    """
    A class for managing a registry of agents.
    """

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

        # Initialize the agent registry
        self.agent_registry = AgentRegistrySchema(
            name=self.name,
            description=self.description,
            agents=[],
            number_of_agents=len(agents) if agents else 0,
        )

        if agents:
            self.add_many(agents)

    def add(self, agent: Agent) -> None:
        """
        Adds a new agent to the registry.

        Raises:
            ValueError: If the agent_name is invalid or already exists in the registry.
            ValidationError: If the input data is invalid.
        """
        name = agent.agent_name

        # ✅ Validation for agent_name
        if not isinstance(name, str) or not name.strip():
            logger.error("Invalid agent_name. It must be a non-empty string.")
            raise ValueError("Invalid agent_name. It must be a non-empty string.")

        self.agent_to_py_model(agent)

        with self.lock:
            if name in self.agents:
                logger.error(f"Agent with name {name} already exists.")
                raise ValueError(f"Agent with name {name} already exists.")
            try:
                self.agents[name] = agent
                logger.info(f"Agent {name} added successfully.")
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise

    def add_many(self, agents: List[Agent]) -> None:
        """
        Adds multiple agents to the registry.
        Stops immediately if any agent has an invalid name.
        """
        # ✅ Pre-validation before threading
        for agent in agents:
            if not isinstance(agent.agent_name, str) or not agent.agent_name.strip():
                logger.error(f"Invalid agent_name in batch: {agent.agent_name!r}")
                raise ValueError(
                    f"Invalid agent_name in batch: {agent.agent_name!r}"
                )

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.add, agent): agent for agent in agents}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error adding agent: {e}")
                    raise

    def delete(self, agent_name: str) -> None:
        with self.lock:
            try:
                del self.agents[agent_name]
                logger.info(f"Agent {agent_name} deleted successfully.")
            except KeyError as e:
                logger.error(f"Error: {e}")
                raise

    def update_agent(self, agent_name: str, new_agent: Agent) -> None:
        with self.lock:
            if agent_name not in self.agents:
                logger.error(f"Agent with name {agent_name} does not exist.")
                raise KeyError(f"Agent with name {agent_name} does not exist.")
            try:
                self.agents[agent_name] = new_agent
                logger.info(f"Agent {agent_name} updated successfully.")
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise

    def get(self, agent_name: str) -> Agent:
        with self.lock:
            try:
                agent = self.agents[agent_name]
                logger.info(f"Agent {agent_name} retrieved successfully.")
                return agent
            except KeyError as e:
                logger.error(f"Error: {e}")
                raise

    def list_agents(self) -> List[str]:
        try:
            with self.lock:
                agent_names = list(self.agents.keys())
                logger.info("Listing all agents.")
                return agent_names
        except Exception as e:
            logger.error(f"Error: {e}")
            raise e

    def return_all_agents(self) -> List[Agent]:
        try:
            with self.lock:
                agents = list(self.agents.values())
                logger.info("Returning all agents.")
                return agents
        except Exception as e:
            logger.error(f"Error: {e}")
            raise e

    def query(self, condition: Optional[Callable[[Agent], bool]] = None) -> List[Agent]:
        try:
            with self.lock:
                if condition is None:
                    agents = list(self.agents.values())
                    logger.info("Querying all agents.")
                    return agents
                agents = [agent for agent in self.agents.values() if condition(agent)]
                logger.info("Querying agents with condition.")
                return agents
        except Exception as e:
            logger.error(f"Error: {e}")
            raise e

    def find_agent_by_name(self, agent_name: str) -> Optional[Agent]:
        try:
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(self.get, agent_name): agent_name
                    for agent_name in self.agents.keys()
                }
                for future in as_completed(futures):
                    agent = future.result()
                    if agent.agent_name == agent_name:
                        return agent
        except Exception as e:
            logger.error(f"Error: {e}")
            raise e

    def find_agent_by_id(self, agent_id: str) -> Optional[Agent]:
        return self.agents.get(agent_id)

    def agents_to_json(self) -> str:
        agents_dict = {name: agent.to_dict() for name, agent in self.agents.items()}
        return json.dumps(agents_dict, indent=4)

    def agent_to_py_model(self, agent: Agent):
        agent_name = agent.agent_name
        agent_description = (
            agent.description if agent.description else "No description provided"
        )

        schema = AgentConfigSchema(
            uuid=agent.id,
            name=agent_name,
            description=agent_description,
            config=agent.to_dict(),
        )

        logger.info(f"Agent {agent_name} converted to Pydantic model.")
        self.agent_registry.agents.append(schema)


# if __name__ == "__main__":
#     from swarms import Agent

#     agent1 = Agent(agent_name="test_agent_1")
#     agent2 = Agent(agent_name="test_agent_2")
#     agent3 = Agent(agent_name="test_agent_3")
#     print(f"Created agents: {agent1}, {agent2}, {agent3}")

#     registry = AgentRegistry()
#     print(f"Created agent registry: {registry}")

#     registry.add(agent1)
#     registry.add(agent2)
#     registry.add(agent3)
#     print(f"Added agents to registry: {agent1}, {agent2}, {agent3}")

#     all_agents = registry.return_all_agents()
#     print(f"All agents in registry: {all_agents}")

#     found_agent1 = registry.find_agent_by_name("test_agent_1")
#     found_agent2 = registry.find_agent_by_name("test_agent_2")
#     found_agent3 = registry.find_agent_by_name("test_agent_3")
#     print(f"Found agents by name: {found_agent1}, {found_agent2}, {found_agent3}")

#     print(registry.agents_to_json())
