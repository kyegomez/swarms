from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from swarms.workers.base import AbstractWorker


class AbstractSwarm(ABC):
    """
    Abstract class for swarm simulation architectures


    Methods:
    ---------

    communicate()
        Communicate with the swarm through the orchestrator, protocols, and the universal communication layer

    run()
        Run the swarm

    arun()
        Run the swarm Asynchronously

    add_worker(worker: "AbstractWorker")
        Add a worker to the swarm

    remove_worker(worker: "AbstractWorker")
        Remove a worker from the swarm

    broadcast(message: str, sender: Optional["AbstractWorker"] = None)
        Broadcast a message to all workers

    reset()
        Reset the swarm

    plan(task: str)
        Workers must individually plan using a workflow or pipeline

    direct_message(message: str, sender: "AbstractWorker", recipient: "AbstractWorker")
        Send a direct message to a worker

    autoscaler(num_workers: int, worker: ["AbstractWorker"])
        Autoscaler that acts like kubernetes for autonomous agents

    get_worker_by_id(id: str) -> "AbstractWorker"
        Locate a worker by id

    get_worker_by_name(name: str) -> "AbstractWorker"
        Locate a worker by name

    assign_task(worker: "AbstractWorker", task: Any) -> Dict
        Assign a task to a worker

    get_all_tasks(worker: "AbstractWorker", task: Any)
        Get all tasks

    get_finished_tasks() -> List[Dict]
        Get all finished tasks

    get_pending_tasks() -> List[Dict]
        Get all pending tasks

    pause_worker(worker: "AbstractWorker", worker_id: str)
        Pause a worker

    resume_worker(worker: "AbstractWorker", worker_id: str)
        Resume a worker

    stop_worker(worker: "AbstractWorker", worker_id: str)
        Stop a worker

    restart_worker(worker: "AbstractWorker")
        Restart worker

    scale_up(num_worker: int)
        Scale up the number of workers

    scale_down(num_worker: int)
        Scale down the number of workers


    """

    # TODO: Pass in abstract LLM class that can utilize Hf or Anthropic models, Move away from OPENAI
    # TODO: ADD Universal Communication Layer, a ocean vectorstore instance
    # TODO: BE MORE EXPLICIT ON TOOL USE, TASK DECOMPOSITION AND TASK COMPLETETION AND ALLOCATION
    # TODO: Add RLHF Data collection, ask user how the swarm is performing
    # TODO: Create an onboarding process if not settings are preconfigured like `from swarms import Swarm, Swarm()` => then initiate onboarding name your swarm + provide purpose + etc

    @abstractmethod
    def __init__(self, workers: List["AbstractWorker"]):
        """Initialize the swarm with workers"""
        pass

    @abstractmethod
    def communicate(self):
        """Communicate with the swarm through the orchestrator, protocols, and the universal communication layer"""
        pass

    @abstractmethod
    def run(self):
        """Run the swarm"""
        pass

    @abstractmethod
    def arun(self):
        """Run the swarm Asynchronously"""
        pass

    @abstractmethod
    def add_worker(self, worker: "AbstractWorker"):
        """Add a worker to the swarm"""
        pass

    @abstractmethod
    def remove_worker(self, worker: "AbstractWorker"):
        """Remove a worker from the swarm"""
        pass

    @abstractmethod
    def broadcast(
        self, message: str, sender: Optional["AbstractWorker"] = None
    ):
        """Broadcast a message to all workers"""
        pass

    @abstractmethod
    def reset(self):
        """Reset the swarm"""
        pass

    @abstractmethod
    def plan(self, task: str):
        """Workers must individually plan using a workflow or pipeline"""
        pass

    @abstractmethod
    def direct_message(
        self,
        message: str,
        sender: "AbstractWorker",
        recipient: "AbstractWorker",
    ):
        """Send a direct message to a worker"""
        pass

    @abstractmethod
    def autoscaler(
        self, num_workers: int, worker: ["AbstractWorker"]
    ):
        """Autoscaler that acts like kubernetes for autonomous agents"""
        pass

    @abstractmethod
    def get_worker_by_id(self, id: str) -> "AbstractWorker":
        """Locate a worker by id"""
        pass

    @abstractmethod
    def get_worker_by_name(self, name: str) -> "AbstractWorker":
        """Locate a worker by name"""
        pass

    @abstractmethod
    def assign_task(
        self, worker: "AbstractWorker", task: Any
    ) -> Dict:
        """Assign a task to a worker"""
        pass

    @abstractmethod
    def get_all_tasks(self, worker: "AbstractWorker", task: Any):
        """Get all tasks"""

    @abstractmethod
    def get_finished_tasks(self) -> List[Dict]:
        """Get all finished tasks"""
        pass

    @abstractmethod
    def get_pending_tasks(self) -> List[Dict]:
        """Get all pending tasks"""
        pass

    @abstractmethod
    def pause_worker(self, worker: "AbstractWorker", worker_id: str):
        """Pause a worker"""
        pass

    @abstractmethod
    def resume_worker(self, worker: "AbstractWorker", worker_id: str):
        """Resume a worker"""
        pass

    @abstractmethod
    def stop_worker(self, worker: "AbstractWorker", worker_id: str):
        """Stop a worker"""
        pass

    @abstractmethod
    def restart_worker(self, worker: "AbstractWorker"):
        """Restart worker"""
        pass

    @abstractmethod
    def scale_up(self, num_worker: int):
        """Scale up the number of workers"""
        pass

    @abstractmethod
    def scale_down(self, num_worker: int):
        """Scale down the number of workers"""
        pass

    @abstractmethod
    def scale_to(self, num_worker: int):
        """Scale to a specific number of workers"""
        pass

    @abstractmethod
    def get_all_workers(self) -> List["AbstractWorker"]:
        """Get all workers"""
        pass

    @abstractmethod
    def get_swarm_size(self) -> int:
        """Get the size of the swarm"""
        pass

    @abstractmethod
    def get_swarm_status(self) -> Dict:
        """Get the status of the swarm"""
        pass

    @abstractmethod
    def save_swarm_state(self):
        """Save the swarm state"""
        pass
