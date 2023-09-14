import logging
import queue
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

import chromadb
from chromadb.utils import embedding_functions


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
        self.agents = queue.Queue()
        
        for _ in range(agent_list):
            self.agents.put(agent())

        self.task_queue = queue.Queue()

        self.chroma_client = chromadb.Client()

        self.collection = self.chroma_client.create_collection(
            name = collection_name
        )

        self.current_tasks = {}

        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.executor = ThreadPoolExecutor(max_workers=len(agent_list))

        
    @abstractmethod
    def assign_task(
        self, 
        agent_id: int, 
        task: Dict[str, Any]
    ) -> None:
        """Assign a task to a specific agent"""

        while True:
            with self.condition:
                while not self.task_queue:
                    self.condition.wait()
                agent = self.agents.get()
                task = self.task_queue.get()
            
            try:
                result, vector_representation = agent.process_task(
                    task
                )
                self.collection.add(
                    embeddings=[vector_representation],
                    documents=[str(id(task))],
                    ids=[str(id(task))]
                )
                logging.info(f"Task {id(str)} has been processed by agent {id(agent)} with")
            
            except Exception as error:
                logging.error(f"Failed to process task {id(task)} by agent {id(agent)}. Error: {error}")
            finally:
                with self.condition:
                    self.agents.put(agent)
                    self.condition.notify()

    def embed(self, input, api_key, model_name):
        openai = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=model_name
        )

        embedding = openai(input)
        # print(embedding)
        
        embedding_metadata = {input: embedding}
        print(embedding_metadata)

        # return embedding
                
    
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

