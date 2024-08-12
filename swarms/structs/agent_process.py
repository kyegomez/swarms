from datetime import datetime

from pydantic import BaseModel

from swarms.structs.omni_agent_types import AgentListType
from swarms.utils.loguru_logger import logger
from typing import Callable


class AgentProcess(BaseModel):
    agent_id: int
    agent_name: str
    prompt: str
    response: str = None
    time: Callable = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    priority: int = 0
    status: str = "Waiting"
    pid: int = None

    def set_pid(self, pid: int):
        self.pid = pid

    def get_pid(self):
        return self.pid

    def set_time(self, time: callable):
        self.time = time

    def get_time(self):
        return self.time


class AgentProcessQueue:
    """
    A class representing a queue of agent processes.

    Attributes:
        MAX_PID (int): The maximum process ID.
        pid_pool (list): A list representing the availability of process IDs.
        agent_process_queue (list): A list representing the queue of agent processes.

    Methods:
        add(agent_process): Adds an agent process to the queue.
        print(): Prints the details of all agent processes in the queue.

    Private Methods:
        _get_available_pid(): Returns an available process ID from the pool.
    """

    def __init__(self, max_pid: int = 1024):
        self.MAX_PID = max_pid
        self.pid_pool = [False for i in range(self.MAX_PID)]
        self.agent_process_queue = (
            []
        )  # Currently use list to simulate queue

    def add(self, agents: AgentListType):
        """
        Adds an agent process to the queue.

        Args:
            agent_process (AgentProcess): The agent process to be added.

        Returns:
            None
        """
        for agent in agents:
            agent_process = AgentProcess(
                agent_id=agent.id,
                agent_name=agent.agent_name,
                prompt=agent.short_memory.return_history_as_string(),
            )
            pid = self._get_available_pid()
            if pid is None:
                logger.warning("No available PID")
                return
            agent_process.set_pid(pid)
            agent_process.set_status("Waiting")
            self.agent_process_queue.append(agent_process)

    def print(self):
        """
        Prints the details of all agent processes in the queue.

        Returns:
            None
        """
        for agent_process in self.agent_process_queue:
            logger.info(
                f"| Agent-process ID: {agent_process.get_pid()} |"
                f" Status: {agent_process.get_status()} |"
            )

    def _get_available_pid(self):
        """
        Returns an available process ID from the pool.

        Returns:
            int or None: The available process ID, or None if no ID is available.
        """
        for i, used in enumerate(self.pid_pool):
            if not used:
                return i
        return None
