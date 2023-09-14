import logging
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, List

# from swarms.memory.ocean import OceanDB

import chromadb

## =========> 
class Orchestrator(ABC):
    def __init__(
        self, 
        agent, 
        agent_list: List[Any], 
        task_queue: List[Any], 
        collection_name: str = "swarm"
    ):
        self.agent = agent
        self.agents = [agent() for _ in range(agent_list)]
        self.task_queue = task_queue

        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection(
            name=collection_name
        )

        self.current_tasks = {}
        self.lock = threading.Lock()
        
    @abstractmethod
    def assign_task(
        self, 
        agent_id: int, 
        task: Dict[str, Any]
    ) -> None:
        """Assign a task to a specific agent"""

        with self.lock:
            if self.task_queue:
                #get and agent and a task
                agent = self.agents.pop(0)
                task = self.task_queue.popleft()
                result, vector_representation = agent.process_task()

                #use chromas's method to add data
                self.collection.add(
                    embeddings=[vector_representation],
                    documents=[str(id(task))],
                    ids=[str(id(task))]
                )

                #put the agent back to agent slist
                self.agents.append(agent)
                logging.info(f"Task {id(str)} has been processed by agent {id(agent)} ")
                return result
            else:
                logging.error("Task queue is empty")
    
    @abstractmethod
    def retrieve_results(self, agent_id: int) -> Any:
        """Retrieve results from a specific agent"""

        try:
            #Query the vector database for documents created by the agents 
            results = self.collection.query(
                query_texts=[str(agent_id)], 
                n_results=10
            )

            return results
        except Exception as e:
            logging.error(f"Failed to retrieve results from agent {agent_id}. Error {e}")
            raise
    
    @abstractmethod
    def update_vector_db(self, data) -> None:
        """Update the vector database"""

        try:
            self.collection.add(
                embeddings=[data["vector"]],
                documents=[str(data["task_id"])],
                ids=[str(data["task_id"])]
            )

        except Exception as e:
            logging.error(f"Failed to update the vector database. Error: {e}")
            raise


    @abstractmethod
    def get_vector_db(self):
        """Retrieve the vector database"""
        return self.collection

    def append_to_db(self, result: str):
        """append the result of the swarm to a specifici collection in the database"""

        try:
            self.collection.add(
                documents=[result],
                ids=[str(id(result))]
            )

        except Exception as e:
            logging.error(f"Failed to append the agent output to database. Error: {e}")
            raise

    def run(self, objective:str):
        """Runs"""
        if not objective or not isinstance(objective, str):
            logging.error("Invalid objective")
            raise ValueError("A valid objective is required")
        
        try:
            self.task_queue.append(objective)
            results = [
                self.assign_task(
                    agent_id, task
                ) for agent_id, task in zip(
                    range(
                        len(self.agents)
                    ), self.task_queue
                )
            ]
            
            for result in results:
                self.append_to_db(result)
            
            logging.info(f"Successfully ran swarms with results: {results}")
            return results
        except Exception as e:
            logging.error(f"An error occured in swarm: {e}")
            return None

