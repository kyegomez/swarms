import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError

from swarms import Agent
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

    Attributes:
        name (str): The name of the registry.
        description (str): A description of the registry.
        return_json (bool): Indicates whether to return data in JSON format.
        auto_save (bool): Indicates whether to automatically save changes to the registry.
        agents (Dict[str, Agent]): A dictionary of agents in the registry, keyed by agent name.
        lock (Lock): A lock for thread-safe operations on the registry.
        agent_registry (AgentRegistrySchema): The schema for the agent registry.
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
        """
        Initializes the AgentRegistry.

        Args:
            name (str, optional): The name of the registry. Defaults to "Agent Registry".
            description (str, optional): A description of the registry. Defaults to "A registry for managing agents.".
            agents (Optional[List[Agent]], optional): A list of agents to initially add to the registry. Defaults to None.
            return_json (bool, optional): Indicates whether to return data in JSON format. Defaults to True.
            auto_save (bool, optional): Indicates whether to automatically save changes to the registry. Defaults to False.
        """
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

        Args:
            agent (Agent): The agent to add.

        Raises:
            ValueError: If the agent_name already exists in the registry.
            ValidationError: If the input data is invalid.
        """
        name = agent.agent_name

        self.agent_to_py_model(agent)

        with self.lock:
            if name in self.agents:
                logger.error(
                    f"Agent with name {name} already exists."
                )
                raise ValueError(
                    f"Agent with name {name} already exists."
                )
            try:
                self.agents[name] = agent
                logger.info(f"Agent {name} added successfully.")
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise

    def add_many(self, agents: List[Agent]) -> None:
        """
        Adds multiple agents to the registry.

        Args:
            agents (List[Agent]): The list of agents to add.

        Raises:
            ValueError: If any of the agent_names already exist in the registry.
            ValidationError: If the input data is invalid.
        """
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.add, agent): agent
                for agent in agents
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error adding agent: {e}")
                    raise

    def delete(self, agent_name: str) -> None:
        """
        Deletes an agent from the registry.

        Args:
            agent_name (str): The name of the agent to delete.

        Raises:
            KeyError: If the agent_name does not exist in the registry.
        """
        with self.lock:
            try:
                del self.agents[agent_name]
                logger.info(
                    f"Agent {agent_name} deleted successfully."
                )
            except KeyError as e:
                logger.error(f"Error: {e}")
                raise

    def update_agent(self, agent_name: str, new_agent: Agent) -> None:
        """
        Updates an existing agent in the registry.

        Args:
            agent_name (str): The name of the agent to update.
            new_agent (Agent): The new agent to replace the existing one.

        Raises:
            KeyError: If the agent_name does not exist in the registry.
            ValidationError: If the input data is invalid.
        """
        with self.lock:
            if agent_name not in self.agents:
                logger.error(
                    f"Agent with name {agent_name} does not exist."
                )
                raise KeyError(
                    f"Agent with name {agent_name} does not exist."
                )
            try:
                self.agents[agent_name] = new_agent
                logger.info(
                    f"Agent {agent_name} updated successfully."
                )
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise

    def get(self, agent_name: str) -> Agent:
        """
        Retrieves an agent from the registry.

        Args:
            agent_name (str): The name of the agent to retrieve.

        Returns:
            Agent: The agent associated with the given agent_name.

        Raises:
            KeyError: If the agent_name does not exist in the registry.
        """
        with self.lock:
            try:
                agent = self.agents[agent_name]
                logger.info(
                    f"Agent {agent_name} retrieved successfully."
                )
                return agent
            except KeyError as e:
                logger.error(f"Error: {e}")
                raise

    def list_agents(self) -> List[str]:
        """
        Lists all agent names in the registry.

        Returns:
            List[str]: A list of all agent names.
        """
        try:
            with self.lock:
                agent_names = list(self.agents.keys())
                logger.info("Listing all agents.")
                return agent_names
        except Exception as e:
            logger.error(f"Error: {e}")
            raise e

    def return_all_agents(self) -> List[Agent]:
        """
        Returns all agents from the registry.

        Returns:
            List[Agent]: A list of all agents.
        """
        try:
            with self.lock:
                agents = list(self.agents.values())
                logger.info("Returning all agents.")
                return agents
        except Exception as e:
            logger.error(f"Error: {e}")
            raise e

    def query(
        self, condition: Optional[Callable[[Agent], bool]] = None
    ) -> List[Agent]:
        """
        Queries agents based on a condition.

        Args:
            condition (Optional[Callable[[Agent], bool]]): A function that takes an agent and returns a boolean indicating
                                                           whether the agent meets the condition.

        Returns:
            List[Agent]: A list of agents that meet the condition.
        """
        try:
            with self.lock:
                if condition is None:
                    agents = list(self.agents.values())
                    logger.info("Querying all agents.")
                    return agents

                agents = [
                    agent
                    for agent in self.agents.values()
                    if condition(agent)
                ]
                logger.info("Querying agents with condition.")
                return agents
        except Exception as e:
            logger.error(f"Error: {e}")
            raise e

    def find_agent_by_name(self, agent_name: str) -> Optional[Agent]:
        """
        Find an agent by its name.

        Args:
            agent_name (str): The name of the agent to find.

        Returns:
            Agent: The agent with the given name.
        """
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

    def agent_to_py_model(self, agent: Agent):
        """
        Converts an agent to a Pydantic model.

        Args:
            agent (Agent): The agent to convert.
        """
        agent_name = agent.agent_name
        agent_description = (
            agent.description
            if agent.description
            else "No description provided"
        )

        schema = AgentConfigSchema(
            uuid=agent.id,
            name=agent_name,
            description=agent_description,
            config=agent.to_dict(),
        )

        logger.info(
            f"Agent {agent_name} converted to Pydantic model."
        )

        self.agent_registry.agents.append(schema)
