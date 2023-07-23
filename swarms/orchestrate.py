from abc import ABC, abstractmethod
import celery
from typing import List, Dict, Any
import numpy as np

class Orchestrator(ABC):
    def __init__(self, agent_list: List[Any], task_queue: celery.Celery, vector_db: np.ndarray):
        self.agents = agent_list
        self.task_queue = task_queue
        self.vector_db = vector_db
        self.current_tasks = {}
        
    @abstractmethod
    def assign_task(self, agent_id: int, task: Dict[str, Any]) -> None:
        """Assign a task to a specific agent"""
        pass
    
    @abstractmethod
    def retrieve_results(self, agent_id: int) -> Any:
        """Retrieve results from a specific agent"""
        pass
    
    @abstractmethod
    def update_vector_db(self, data: np.ndarray) -> None:
        """Update the vector database"""
        pass

    @abstractmethod
    def get_vector_db(self) -> np.ndarray:
        """Retrieve the vector database"""
        pass
