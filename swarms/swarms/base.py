from abc import ABC, abstractmethod
from typing import Optional, List
from swarms.workers.base import AbstractWorker

class AbstractSwarm(ABC):
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
    def broadcast(self, message: str, sender: Optional["AbstractWorker"] = None):
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
