# 10 improvements to the `Orchestrator` class to enable more flexibility and usability:

1.  Dynamic Agent Creation: Allow the number of agents to be specified at runtime, rather than being fixed at the time of instantiation.

```
def add_agents(self, num_agents: int):
    for _ in range(num_agents):
        self.agents.put(self.agent())
    self.executor = ThreadPoolExecutor(max_workers=self.agents.qsize())
```

1.  Agent Removal: Allow agents to be removed from the pool.

```
def remove_agents(self, num_agents: int):
    for _ in range(num_agents):
        if not self.agents.empty():
            self.agents.get()
    self.executor = ThreadPoolExecutor(max_workers=self.agents.qsize())
```

1.  Task Prioritization: Allow tasks to be prioritized.

```
from queue import PriorityQueue

def __init__(self, agent, agent_list: List[Any], task_queue: List[Any], collection_name: str = "swarm", api_key: str = None, model_name: str = None):
    # ...
    self.task_queue = PriorityQueue()
    # ...

def add_task(self, task: Dict[str, Any], priority: int = 0):
    self.task_queue.put((priority, task))
```

1.  Task Status: Track the status of tasks.

```
from enum import Enum

class TaskStatus(Enum):
    QUEUED = 1
    RUNNING = 2
    COMPLETED = 3
    FAILED = 4

# In assign_task method
self.current_tasks[id(task)] = TaskStatus.RUNNING
# On successful completion
self.current_tasks[id(task)] = TaskStatus.COMPLETED
# On failure
self.current_tasks[id(task)] = TaskStatus.FAILED
```

1.  Result Retrieval: Allow results to be retrieved by task ID.

```
def retrieve_result(self, task_id: int) -> Any:
    return self.collection.query(query_texts=[str(task_id)], n_results=1)
```

1.  Batch Task Assignment: Allow multiple tasks to be assigned at once.

```
def assign_tasks(self, tasks: List[Dict[str, Any]]):
    for task in tasks:
        self.task_queue.put(task)
```

1.  Error Handling: Improve error handling by re-queuing failed tasks.

```
# In assign_task method
except Exception as error:
    logging.error(f"Failed to process task {id(task)} by agent {id(agent)}. Error: {error}")
    self.task_queue.put(task)
```

1.  Agent Status: Track the status of agents (e.g., idle, working).

```
self.agent_status = {id(agent): "idle" for agent in self.agents.queue}

# In assign_task method
self.agent_status[id(agent)] = "working"
# On task completion
self.agent_status[id(agent)] = "idle"
```

1.  Custom Embedding Function: Allow a custom embedding function to be used.

```
def __init__(self, agent, agent_list: List[Any], task_queue: List[Any], collection_name: str = "swarm", api_key: str = None, model_name: str = None, embed_func=None):
    # ...
    self.embed_func = embed_func if embed_func else self.embed
    # ...

def embed(self, input, api_key, model_name):
    # ...
    embedding = self.embed_func(input)
    # ...
```

1.  Agent Communication: Allow agents to communicate with each other.

```
def communicate(self, sender_id: int, receiver_id: int, message: str):
    message_vector = self.embed_func(message)
    self.collection.add(embeddings=[message_vector], documents=[message], ids=[f"{sender_id}_to_{receiver_id}"])
```



```
import logging
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List
from enum import Enum

import chromadb
from chromadb.utils import embedding_functions

class TaskStatus(Enum):
    QUEUED = 1
    RUNNING = 2
    COMPLETED = 3
    FAILED = 4

class Orchestrator:
    def __init__(self, agent, agent_list: List[Any], task_queue: List[Any], collection_name: str = "swarm", api_key: str = None, model_name: str = None, embed_func=None):
        self.agent = agent
        self.agents = queue.Queue()
        self.agent_status = {}

        self.add_agents(agent_list)

        self.task_queue = queue.PriorityQueue()

        self.chroma_client = chromadb.Client()

        self.collection = self.chroma_client.create_collection(name = collection_name)

        self.current_tasks = {}

        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)

        self.embed_func = embed_func if embed_func else self.embed

    def add_agents(self, num_agents: int):
        for _ in range(num_agents):
            agent = self.agent()
            self.agents.put(agent)
            self.agent_status[id(agent)] = "idle"
        self.executor = ThreadPoolExecutor(max_workers=self.agents.qsize())

    def remove_agents(self, num_agents: int):
        for _ in range(num_agents):
            if not self.agents.empty():
                agent = self.agents.get()
                del self.agent_status[id(agent)]
        self.executor = ThreadPoolExecutor(max_workers=self.agents.qsize())

    def assign_task(self, agent_id: int, task: Dict[str, Any]) -> None:
        while True:
            with self.condition:
                while not self.task_queue:
                    self.condition.wait()
                agent = self.agents.get()
                task = self.task_queue.get()

            try:
                self.agent_status[id(agent)] = "working"
                result = self.worker.run(task["content"])

                vector_representation = self.embed_func(result)

                self.collection.add(embeddings=[vector_representation], documents=[str(id(task))], ids=[str(id(task))])

                logging.info(f"Task {id(str)} has been processed by agent {id(agent)} with")
                self.current_tasks[id(task)] = TaskStatus.COMPLETED

            except Exception as error:
                logging.error(f"Failed to process task {id(task)} by agent {id(agent)}. Error: {error}")
                self.current_tasks[id(task)] = TaskStatus.FAILED
                self.task_queue.put(task)
            finally:
                with self.condition:
                    self.agent_status[id(agent)] = "idle"
                    self.agents.put(agent)
                    self.condition.notify()

    def embed(self, input):
        openai = embedding_functions.OpenAIEmbeddingFunction(api_key=self.api_key, model_name=self.model_name)
        embedding = openai(input)
        return embedding

    def retrieve_results(self, agent_id: int) -> Any:
        try:
            results = self.collection.query(query_texts=[str(agent_id)], n_results=10)
            return results
        except Exception as e:
            logging.error(f"Failed to retrieve results from agent {id(agent_id)}. Error {e}")
            raise

    def update_vector_db(self, data) -> None:
        try:
            self.collection.add(embeddings=[data["vector"]], documents=[str(data["task_id"])], ids=[str(data["task_id"])])
        except Exception as e:
            logging.error(f"Failed to update the vector database. Error: {e}")
            raise

    def get_vector_db(self):
        return self.collection

    def append_to_db(self, result: str):
        try:
            self.collection.add(documents=[result], ids=[str(id(result))])
        except Exception as e:
            logging.error(f"Failed to append the agent output to database. Error: {e}")
            raise

    def run(self, objective:str):
        if not objective or not isinstance(objective, str):
            logging.error("Invalid objective")
            raise ValueError("A valid objective is required")

        try:
            self.task_queue.put((0, objective))

            results = [self.assign_task(agent_id, task) for agent_id, task in zip(range(len(self.agents)), self.task_queue)]

            for result in results:
                self.append_to_db(result)

            logging.info(f"Successfully ran swarms with results: {results}")
            return results
        except Exception as e:
            logging.error(f"An error occured in swarm: {e}")
            return None

    def chat(self, sender_id: int, receiver_id: int, message: str):
        message_vector = self.embed_func(message)

        # Store the message in the vector database
        self.collection.add(embeddings=[message_vector], documents=[message], ids=[f"{sender_id}_to_{receiver_id}"])

    def assign_tasks(self, tasks: List[Dict[str, Any]], priority: int = 0):
        for task in tasks:
            self.task_queue.put((priority, task))

    def retrieve_result(self, task_id: int) -> Any:
        try:
            result = self.collection.query(query_texts=[str(task_id)], n_results=1)
            return result
        except Exception as e:
            logging.error(f"Failed to retrieve result for task {task_id}. Error: {e}")
            raise
```

With these improvements, the `Orchestrator` class now supports dynamic agent creation and removal, task prioritization, task status tracking, result retrieval by task ID, batch task assignment, improved error handling, agent status tracking, custom embedding functions, and agent communication. This should make the class more flexible and easier to use when creating swarms of LLMs.