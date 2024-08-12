"""
Todo
- [ ] Test the new api feature
- [ ] Add the agent schema for every agent -- following OpenAI assistaants schema
- [ ] then add the swarm schema for the swarm url: /v1/swarms/{swarm_name}/agents/{agent_id}
- [ ] Add the agent schema for the agent url: /v1/swarms/{swarm_name}/agents/{agent_id}
"""

import asyncio
import multiprocessing
import queue
import threading
from typing import List, Optional

import tenacity

# from fastapi import FastAPI
from pydantic import BaseModel

from swarms.structs.agent import Agent
from swarms.structs.base_swarm import BaseSwarm
from swarms.utils.loguru_logger import logger


# Pydantic models
class TaskRequest(BaseModel):
    task: str


# Pydantic models
class TaskResponse(BaseModel):
    result: str


class AgentInfo(BaseModel):
    agent_name: str
    agent_description: str


class SwarmInfo(BaseModel):
    swarm_name: str
    swarm_description: str
    agents: List[AgentInfo]


# Helper function to get the number of workers
def get_number_of_workers():
    return multiprocessing.cpu_count()


# [TODO] Add the agent schema for every agent -- following OpenAI assistaants schema
class SwarmNetwork(BaseSwarm):
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
        name: str = None,
        description: str = None,
        agents: List[Agent] = None,
        idle_threshold: float = 0.2,
        busy_threshold: float = 0.7,
        api_enabled: Optional[bool] = False,
        logging_enabled: Optional[bool] = False,
        api_on: Optional[bool] = False,
        host: str = "0.0.0.0",
        port: int = 8000,
        swarm_callable: Optional[callable] = None,
        *args,
        **kwargs,
    ):
        super().__init__(agents=agents, *args, **kwargs)
        self.name = name
        self.description = description
        self.agents = agents
        self.task_queue = queue.Queue()
        self.idle_threshold = idle_threshold
        self.busy_threshold = busy_threshold
        self.lock = threading.Lock()
        self.api_enabled = api_enabled
        self.logging_enabled = logging_enabled
        self.host = host
        self.port = port
        self.swarm_callable = swarm_callable

        # Ensure that the agents list is not empty
        if not agents:
            raise ValueError("The agents list cannot be empty")

        # Create a dictionary of agents for easy access
        self.agent_dict = {agent.id: agent for agent in agents}

        # # Create the FastAPI instance
        # if api_on is True:
        #     logger.info("Creating FastAPI instance")
        #     self.app = FastAPI(debug=True, *args, **kwargs)

        #     self.app.add_middleware(
        #         CORSMiddleware,
        #         allow_origins=["*"],
        #         allow_credentials=True,
        #         allow_methods=["*"],
        #         allow_headers=["*"],
        #     )

        #     logger.info("Routes set for creation")
        #     self._create_routes()

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
        self.logger.info(f"Adding task {task} to queue asynchronously")
        try:
            # Add task to queue asynchronously with asyncio
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.task_queue.put, task)
            self.logger.info(f"Task {task} added to queue")
        except Exception as error:
            print(
                f"Error adding task to queue: {error} try again with"
                " a new task"
            )
            raise error

    # def _create_routes(self) -> None:
    #     """
    #     Creates the routes for the API.
    #     """
    #     # Extensive logginbg
    #     logger.info("Creating routes for the API")

    #     # Routes available
    #     logger.info(
    #         "Routes available: /v1/swarms, /v1/health, /v1/swarms/{swarm_name}/agents/{agent_id}, /v1/swarms/{swarm_name}/run"
    #     )

    #     @self.app.get("/v1/swarms", response_model=SwarmInfo)
    #     async def get_swarms() -> SwarmInfo:
    #         try:
    #             logger.info("Getting swarm information")
    #             return SwarmInfo(
    #                 swarm_name=self.swarm_name,
    #                 swarm_description=self.swarm_description,
    #                 agents=[
    #                     AgentInfo(
    #                         agent_name=agent.agent_name,
    #                         agent_description=agent.agent_description,
    #                     )
    #                     for agent in self.agents
    #                 ],
    #             )
    #         except Exception as e:
    #             logger.error(f"Error getting swarm information: {str(e)}")
    #             raise HTTPException(
    #                 status_code=500, detail="Internal Server Error"
    #             )

    #     @self.app.get("/v1/health")
    #     async def get_health() -> Dict[str, str]:
    #         try:
    #             logger.info("Checking health status")
    #             return {"status": "healthy"}
    #         except Exception as e:
    #             logger.error(f"Error checking health status: {str(e)}")
    #             raise HTTPException(
    #                 status_code=500, detail="Internal Server Error"
    #             )

    #     @self.app.get(f"/v1/swarms/{self.swarm_name}/agents/{{agent_id}}")
    #     async def get_agent_info(agent_id: str) -> AgentInfo:
    #         try:
    #             logger.info(f"Getting information for agent {agent_id}")
    #             agent = self.agent_dict.get(agent_id)
    #             if not agent:
    #                 raise HTTPException(
    #                     status_code=404, detail="Agent not found"
    #                 )
    #             return AgentInfo(
    #                 agent_name=agent.agent_name,
    #                 agent_description=agent.agent_description,
    #             )
    #         except Exception as e:
    #             logger.error(f"Error getting agent information: {str(e)}")
    #             raise HTTPException(
    #                 status_code=500, detail="Internal Server Error"
    #             )

    #     @self.app.post(
    #         f"/v1/swarms/{self.swarm_name}/agents/{{agent_id}}/run",
    #         response_model=TaskResponse,
    #     )
    #     async def run_agent_task(
    #         task_request: TaskRequest,
    #     ) -> TaskResponse:
    #         try:
    #             logger.info("Running agent task")
    #             # Assuming only one agent in the swarm for this example
    #             agent = self.agents[0]
    #             logger.info(f"Running agent task: {task_request.task}")
    #             result = agent.run(task_request.task)
    #             return TaskResponse(result=result)
    #         except Exception as e:
    #             logger.error(f"Error running agent task: {str(e)}")
    #             raise HTTPException(
    #                 status_code=500, detail="Internal Server Error"
    #             )

    # def get_app(self) -> FastAPI:
    #     """
    #     Returns the FastAPI instance.

    #     Returns:
    #         FastAPI: The FastAPI instance.
    #     """
    #     return self.app

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
            for agent in self.agents:
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
                agent.run(task, *args, **kwargs) for agent in self.agents
            ]
        except Exception as error:
            logger.error(f"Error running task on agents: {error}")
            raise error

    def list_agents(self):
        """List all agents."""
        self.logger.info("[Listing all active agents]")

        try:
            # Assuming self.agents is a list of agent objects
            for agent in self.agents:
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
            for agent in self.agents:
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
            self.agents.append(agent)
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
            for agent in self.agents:
                if agent.id == agent_id:
                    self.agents.remove(agent)
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
            await loop.run_in_executor(None, self.remove_agent, agent_id)
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
                self.agents.append(Agent())
        except Exception as error:
            print(f"Error scaling up agent pool: {error}")
            raise error

    def scale_down(self, num_agents: int = 1):
        """Scale down the agent pool

        Args:
            num_agents (int, optional): _description_. Defaults to 1.
        """
        for _ in range(num_agents):
            self.agents.pop()

    @tenacity.retry(
        wait=tenacity.wait_fixed(1),
        stop=tenacity.stop_after_attempt(3),
        retry=tenacity.retry_if_exception_type(Exception),
    )
    def run(self, *args, **kwargs):
        """run the swarm network"""
        app = self.get_app()

        try:
            import uvicorn

            logger.info(
                f"Running the swarm network with {len(self.agents)} on {self.host}:{self.port}"
            )
            uvicorn.run(
                app,
                host=self.host,
                port=self.port,
                # workers=get_number_of_workers(),
                *args,
                **kwargs,
            )

            return app
        except Exception as error:
            logger.error(f"Error running the swarm network: {error}")
            raise error


# # # Example usage
# if __name__ == "__main__":

#     agent1 = Agent(
#         agent_name="Covid-19-Chat",
#         agent_description="This agent provides information about COVID-19 symptoms.",
#         llm=OpenAIChat(),
#         max_loops="auto",
#         autosave=True,
#         verbose=True,
#         stopping_condition="finish",
#     )

#     agents = [agent1]  # Add more agents as needed
#     swarm_name = "HealthSwarm"
#     swarm_description = (
#         "A swarm of agents providing health-related information."
#     )

#     agent_api = SwarmNetwork(swarm_name, swarm_description, agents)
#     agent_api.run()
