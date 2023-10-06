from typing import List, Dict, Any, Union
from concurrent.futures import Executor, ThreadPoolExecutor, as_completed
from graphlib import TopologicalSorter

class Task:
    def __init__(
        self,
        id: str, 
        parents: List["Task"] = None,
        children: List["Task"] = None
    ):
        self.id = id
        self.parents = parents
        self.children = children
    
    def can_execute(self):
        raise NotImplementedError

    def execute(self):
        raise NotImplementedError
    
