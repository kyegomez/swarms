#input agent or multiple: => it handles multi agent communication, it handles task assignment, task execution, report back with a status, auto scaling,  number of agent nodes, 
"""
from swarms import WorkerNode, Orchestrate

Orchestrate(WorkerNode, autoscale=True, nodes=int, swarm_type="flat")

"""
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



#PRE CONFIGURED AGENTS WITH domain explicit TOOLS
#Build your own Agent
# Learn from previous runs in session management => it's a sucessful run => omniversal memory for all swarms  