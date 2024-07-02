from typing import List, Dict, Optional, Callable
from pydantic import BaseModel, ValidationError
from threading import Lock
from swarms import Agent
from swarms.utils.loguru_logger import logger
from swarms.utils.report_error_loguru import report_error


class AgentModel(BaseModel):
    """
    Pydantic model for an Agent.
    """

    agent_id: str
    agent: Agent


class AgentRegistry:
    """
    A registry for managing agents, with methods to add, delete, update, and query agents.
    """

    def __init__(self):
        self.agents: Dict[str, AgentModel] = {}
        self.lock = Lock()

    def add(self, agent_id: str, agent: Agent) -> None:
        """
        Adds a new agent to the registry.

        Args:
            agent_id (str): The unique identifier for the agent.
            agent (Agent): The agent to add.

        Raises:
            ValueError: If the agent_id already exists in the registry.
            ValidationError: If the input data is invalid.
        """
        with self.lock:
            if agent_id in self.agents:
                logger.error(f"Agent with id {agent_id} already exists.")
                raise ValueError(
                    f"Agent with id {agent_id} already exists."
                )
            try:
                self.agents[agent_id] = AgentModel(
                    agent_id=agent_id, agent=agent
                )
                logger.info(f"Agent {agent_id} added successfully.")
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise

    def delete(self, agent_id: str) -> None:
        """
        Deletes an agent from the registry.

        Args:
            agent_id (str): The unique identifier for the agent to delete.

        Raises:
            KeyError: If the agent_id does not exist in the registry.
        """
        with self.lock:
            try:
                del self.agents[agent_id]
                logger.info(f"Agent {agent_id} deleted successfully.")
            except KeyError as e:
                logger.error(f"Error: {e}")
                raise

    def update_agent(self, agent_id: str, new_agent: Agent) -> None:
        """
        Updates an existing agent in the registry.

        Args:
            agent_id (str): The unique identifier for the agent to update.
            new_agent (Agent): The new agent to replace the existing one.

        Raises:
            KeyError: If the agent_id does not exist in the registry.
            ValidationError: If the input data is invalid.
        """
        with self.lock:
            if agent_id not in self.agents:
                logger.error(f"Agent with id {agent_id} does not exist.")
                raise KeyError(f"Agent with id {agent_id} does not exist.")
            try:
                self.agents[agent_id] = AgentModel(
                    agent_id=agent_id, agent=new_agent
                )
                logger.info(f"Agent {agent_id} updated successfully.")
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise

    def get(self, agent_id: str) -> Agent:
        """
        Retrieves an agent from the registry.

        Args:
            agent_id (str): The unique identifier for the agent to retrieve.

        Returns:
            Agent: The agent associated with the given agent_id.

        Raises:
            KeyError: If the agent_id does not exist in the registry.
        """
        with self.lock:
            try:
                agent = self.agents[agent_id].agent
                logger.info(f"Agent {agent_id} retrieved successfully.")
                return agent
            except KeyError as e:
                logger.error(f"Error: {e}")
                raise

    def list_agents(self) -> List[str]:
        """
        Lists all agent identifiers in the registry.

        Returns:
            List[str]: A list of all agent identifiers.
        """
        try:
            with self.lock:
                agent_ids = list(self.agents.keys())
                logger.info("Listing all agents.")
                return agent_ids
        except Exception as e:
            report_error(e)
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
                    agents = [
                        agent_model.agent
                        for agent_model in self.agents.values()
                    ]
                    logger.info("Querying all agents.")
                    return agents

                agents = [
                    agent_model.agent
                    for agent_model in self.agents.values()
                    if condition(agent_model.agent)
                ]
                logger.info("Querying agents with condition.")
                return agents
        except Exception as e:
            report_error(e)
            raise e

    def find_agent_by_name(self, agent_name: str) -> Agent:
        try:
            for agent_model in self.agents.values():
                if agent_model.agent.agent_name == agent_name:
                    return agent_model.agent
        except Exception as e:
            report_error(e)
            raise e
