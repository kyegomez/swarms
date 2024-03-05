import asyncio
import logging
import queue
import threading
from typing import List, Optional

from fastapi import FastAPI

from swarms.structs.agent import Agent
from swarms.structs.base import BaseStructure
from swarms.utils.logger import logger  # noqa: F401


class SwarmNetwork(BaseStructure):
    """
    SwarmNetwork class

    The SwarmNetwork class is responsible for managing the agents pool
    and the task queue. It also monitors the health of the agents and
    scales the pool up or down based on the number of pending tasks
    and the current load of the agents.

    For example, if the number of pending tasks is greater than the
    number of agents in the pool, the SwarmNetwork will scale up the
    pool by adding new agents. If the number of pending tasks is less
    than the number of agents in the pool, the SwarmNetwork will scale
    down the pool by removing agents.

    The SwarmNetwork class also provides a simple API for interacting
    with the agents pool. The API is implemented using the Flask
    framework and is enabled by default. The API can be disabled by
    setting the `api_enabled` parameter to False.

    Features:
        - Agent pool management
        - Task queue management
        - Agent health monitoring
        - Agent pool scaling
        - Simple API for interacting with the agent pool
        - Simple API for interacting with the task queue
        - Simple API for interacting with the agent health monitor
        - Simple API for interacting with the agent pool scaler
        - Create APIs for each agent in the pool (optional)
        - Run each agent on it's own thread
        - Run each agent on it's own process
        - Run each agent on it's own container
        - Run each agent on it's own machine
        - Run each agent on it's own cluster


    Attributes:
        task_queue (queue.Queue): A queue for storing tasks.
        idle_threshold (float): The idle threshold for the agents.
        busy_threshold (float): The busy threshold for the agents.
        agents (List[Agent]): A list of agents in the pool.
        api_enabled (bool): A flag to enable/disable the API.
        logging_enabled (bool): A flag to enable/disable logging.

    Example:
        >>> from swarms.structs.agent import Agent
        >>> from swarms.structs.swarm_net import SwarmNetwork
        >>> agent = Agent()
        >>> swarm = SwarmNetwork(agents=[agent])
        >>> swarm.add_task("task")
        >>> swarm.run()

    """

    def __init__(
        self,
        agents: List[Agent] = None,
        idle_threshold: float = 0.2,
        busy_threshold: float = 0.7,
        api_enabled: Optional[bool] = False,
        logging_enabled: Optional[bool] = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.agents = agents
        self.task_queue = queue.Queue()
        self.idle_threshold = idle_threshold
        self.busy_threshold = busy_threshold
        self.lock = threading.Lock()
        self.api_enabled = api_enabled
        self.logging_enabled = logging_enabled
        self.agent_pool = []

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        if api_enabled:
            self.api = FastAPI()

        # For each agent in the pool, run it on it's own thread
        if agents is not None:
            for agent in agents:
                self.agent_pool.append(agent)

    def add_task(self, task):
        """Add task to the task queue

        Args:
            task (_type_): _description_

        Example:
        >>> from swarms.structs.agent import Agent
        >>> from swarms.structs.swarm_net import SwarmNetwork
        >>> agent = Agent()
        >>> swarm = SwarmNetwork(agents=[agent])
        >>> swarm.add_task("task")
        """
        self.logger.info(f"Adding task {task} to queue")
        try:
            self.task_queue.put(task)
            self.logger.info(f"Task {task} added to queue")
        except Exception as error:
            print(
                f"Error adding task to queue: {error} try again with"
                " a new task"
            )
            raise error

    async def async_add_task(self, task):
        """Add task to the task queue

        Args:
            task (_type_): _description_

        Example:
        >>> from swarms.structs.agent import Agent
        >>> from swarms.structs.swarm_net import SwarmNetwork
        >>> agent = Agent()
        >>> swarm = SwarmNetwork(agents=[agent])
        >>> swarm.add_task("task")

        """
        self.logger.info(
            f"Adding task {task} to queue asynchronously"
        )
        try:
            # Add task to queue asynchronously with asyncio
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, self.task_queue.put, task
            )
            self.logger.info(f"Task {task} added to queue")
        except Exception as error:
            print(
                f"Error adding task to queue: {error} try again with"
                " a new task"
            )
            raise error

    def run_single_agent(
        self, agent_id, task: Optional[str], *args, **kwargs
    ):
        """Run agent the task on the agent id

        Args:
            agent_id (_type_): _description_
            task (str, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        self.logger.info(f"Running task {task} on agent {agent_id}")
        try:
            for agent in self.agent_pool:
                if agent.id == agent_id:
                    out = agent.run(task, *args, **kwargs)
            return out
        except Exception as error:
            self.logger.error(f"Error running task on agent: {error}")
            raise error

    def run_many_agents(
        self, task: Optional[str] = None, *args, **kwargs
    ) -> List:
        """Run the task on all agents

        Args:
            task (str, optional): _description_. Defaults to None.

        Returns:
            List: _description_
        """
        self.logger.info(f"Running task {task} on all agents")
        try:
            return [
                agent.run(task, *args, **kwargs)
                for agent in self.agent_pool
            ]
        except Exception as error:
            logger.error(f"Error running task on agents: {error}")
            raise error

    def list_agents(self):
        """List all agents."""
        self.logger.info("[Listing all active agents]")

        try:
            # Assuming self.agent_pool is a list of agent objects
            for agent in self.agent_pool:
                self.logger.info(
                    f"[Agent] [ID: {agent.id}] [Name:"
                    f" {agent.agent_name}] [Description:"
                    f" {agent.agent_description}] [Status: Running]"
                )
        except Exception as error:
            self.logger.error(f"Error listing agents: {error}")
            raise

    def get_agent(self, agent_id):
        """Get agent by id

        Args:
            agent_id (_type_): _description_

        Returns:
            _type_: _description_
        """
        self.logger.info(f"Getting agent {agent_id}")

        try:
            for agent in self.agent_pool:
                if agent.id == agent_id:
                    return agent
            raise ValueError(f"No agent found with ID {agent_id}")
        except Exception as error:
            self.logger.error(f"Error getting agent: {error}")
            raise error

    def add_agent(self, agent: Agent):
        """Add agent to the agent pool

        Args:
            agent (_type_): _description_
        """
        self.logger.info(f"Adding agent {agent} to pool")
        try:
            self.agent_pool.append(agent)
        except Exception as error:
            print(f"Error adding agent to pool: {error}")
            raise error

    def remove_agent(self, agent_id):
        """Remove agent from the agent pool

        Args:
            agent_id (_type_): _description_
        """
        self.logger.info(f"Removing agent {agent_id} from pool")
        try:
            for agent in self.agent_pool:
                if agent.id == agent_id:
                    self.agent_pool.remove(agent)
                    return
            raise ValueError(f"No agent found with ID {agent_id}")
        except Exception as error:
            print(f"Error removing agent from pool: {error}")
            raise error

    async def async_remove_agent(self, agent_id):
        """Remove agent from the agent pool

        Args:
            agent_id (_type_): _description_
        """
        self.logger.info(f"Removing agent {agent_id} from pool")
        try:
            # Remove agent from pool asynchronously with asyncio
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None, self.remove_agent, agent_id
            )
        except Exception as error:
            print(f"Error removing agent from pool: {error}")
            raise error

    def scale_up(self, num_agents: int = 1):
        """Scale up the agent pool

        Args:
            num_agents (int, optional): _description_. Defaults to 1.
        """
        self.logger.info(f"Scaling up agent pool by {num_agents}")
        try:
            for _ in range(num_agents):
                self.agent_pool.append(Agent())
        except Exception as error:
            print(f"Error scaling up agent pool: {error}")
            raise error

    def scale_down(self, num_agents: int = 1):
        """Scale down the agent pool

        Args:
            num_agents (int, optional): _description_. Defaults to 1.
        """
        for _ in range(num_agents):
            self.agent_pool.pop()

    # - Create APIs for each agent in the pool (optional) with fastapi
    def create_apis_for_agents(self):
        """Create APIs for each agent in the pool (optional) with fastapi

        Returns:
            _type_: _description_
        """
        self.apis = []
        for agent in self.agent_pool:
            self.api.get(f"/{agent.id}")

            def run_agent(task: str, *args, **kwargs):
                return agent.run(task, *args, **kwargs)

            self.apis.append(self.api)

    def run(self):
        """run the swarm network"""
        # Observe all agents in the pool
        self.logger.info("Starting the SwarmNetwork")

        for agent in self.agent_pool:
            self.logger.info(f"Starting agent {agent.id}")
            self.logger.info(
                f"[Agent][{agent.id}] [Status] [Running] [Awaiting"
                " Task]"
            )
