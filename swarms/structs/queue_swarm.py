import queue
import threading
from typing import List
from swarms.structs.agent import Agent
import datetime
from pydantic import BaseModel
import os
from swarms.utils.file_processing import create_file_in_folder
from swarms.utils.loguru_logger import logger

time = datetime.datetime.now().isoformat()


class AgentOutput(BaseModel):
    agent_name: str
    task: str
    result: str
    timestamp: str


class SwarmRunMetadata(BaseModel):
    run_id: str
    name: str
    description: str
    agents: List[str]
    start_time: str
    end_time: str
    tasks_completed: int
    outputs: List[AgentOutput]


class TaskQueueSwarm:
    """
    A swarm that processes tasks from a queue using multiple agents on different threads.

    Args:
        agents (List[Agent]): A list of agents of class Agent.
        name (str, optional): The name of the swarm. Defaults to "Task-Queue-Swarm".
        description (str, optional): The description of the swarm. Defaults to "A swarm that processes tasks from a queue using multiple agents on different threads.".
        autosave_on (bool, optional): Whether to automatically save the swarm metadata. Defaults to True.
        save_file_path (str, optional): The file path to save the swarm metadata. Defaults to "swarm_run_metadata.json".
        workspace_dir (str, optional): The directory path of the workspace. Defaults to os.getenv("WORKSPACE_DIR").
        return_metadata_on (bool, optional): Whether to return the swarm metadata after running. Defaults to False.
        max_loops (int, optional): The maximum number of loops to run the swarm. Defaults to 1.

    Attributes:
        name (str): The name of the swarm.
        description (str): The description of the swarm.
        agents (List[Agent]): A list of agents of class Agent.
        task_queue (Queue): A queue to store the tasks.
        lock (Lock): A lock for thread synchronization.
        autosave_on (bool): Whether to automatically save the swarm metadata.
        save_file_path (str): The file path to save the swarm metadata.
        workspace_dir (str): The directory path of the workspace.
        return_metadata_on (bool): Whether to return the swarm metadata after running.
        max_loops (int): The maximum number of loops to run the swarm.
        metadata (SwarmRunMetadata): The metadata of the swarm run.

    """

    def __init__(
        self,
        agents: List[Agent],
        name: str = "Task-Queue-Swarm",
        description: str = "A swarm that processes tasks from a queue using multiple agents on different threads.",
        autosave_on: bool = True,
        save_file_path: str = "swarm_run_metadata.json",
        workspace_dir: str = os.getenv("WORKSPACE_DIR"),
        return_metadata_on: bool = False,
        max_loops: int = 1,
        *args,
        **kwargs,
    ):
        self.name = name
        self.description = description
        self.agents = agents
        self.task_queue = queue.Queue()
        self.lock = threading.Lock()
        self.autosave_on = autosave_on
        self.save_file_path = save_file_path
        self.workspace_dir = workspace_dir
        self.return_metadata_on = return_metadata_on
        self.max_loops = max_loops

        # Metadata
        self.metadata = SwarmRunMetadata(
            run_id=f"swarm_run_{time}",
            name=name,
            description=description,
            agents=[agent.agent_name for agent in agents],
            start_time=time,
            end_time="",
            tasks_completed=0,
            outputs=[],
        )

    def reliability_checks(self):
        logger.info("Initializing reliability checks. ")

        if self.agents is None:
            raise ValueError(
                "You must provide a list of agents of class Agent into the class"
            )

        if self.max_loops == 0:
            raise ValueError(
                "Max loops cannot be zero, the loop must run once to use the swarm"
            )

        logger.info(
            "Reliability checks successful, your swarm is ready for usage."
        )

    def add_task(self, task: str):
        """Adds a task to the queue."""
        self.task_queue.put(task)

    def _process_task(self, agent: Agent):
        """Processes tasks from the queue using the provided agent."""
        while not self.task_queue.empty():
            task = self.task_queue.get()
            try:
                logger.info(
                    f"Agent {agent.agent_name} is running task: {task}"
                )
                result = agent.run(task)
                self.metadata.tasks_completed += 1
                self.metadata.outputs.append(
                    AgentOutput(
                        agent_name=agent.agent_name,
                        task=task,
                        result=result,
                        timestamp=time,
                    )
                )
                logger.info(
                    f"Agent {agent.agent_name} completed task: {task}"
                )
                logger.info(f"Result: {result}")
            except Exception as e:
                logger.error(
                    f"Agent {agent.agent_name} failed to complete task: {task}"
                )
                logger.error(f"Error: {e}")
            self.task_queue.task_done()

    def run(self):
        """Runs the swarm by having agents pick up tasks from the queue."""
        logger.info(f"Starting swarm run: {self.metadata.run_id}")

        threads = []
        for agent in self.agents:
            thread = threading.Thread(
                target=self._process_task, args=(agent,)
            )
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        if self.autosave_on is True:
            self.save_json_to_file()

        if self.return_metadata_on is True:
            return self.metadata

    def save_json_to_file(self):
        json_string = self.export_metadata()

        # Create a file in the current directory
        create_file_in_folder(
            self.workspace_dir, self.save_file_path, json_string
        )

        logger.info(f"Metadata saved to {self.save_file_path}")

        return None

    def export_metadata(self):
        return self.metadata.model_dump_json(indent=4)
