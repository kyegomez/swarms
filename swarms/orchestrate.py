#input agent or multiple: => it handles multi agent communication, it handles task assignment, task execution, report back with a status, auto scaling,  number of agent nodes,
#make it optional to have distributed communication protocols, trco, rdp, http, microsoervice, make it optional to collect data from users runs
"""
# Orchestrator
* Takes in an agent class with vector store, then handles all the communication and scales up a swarm with number of agents and handles task assignment and task completion

```python

from swarms import OpenAI, Orchestrator, Swarm

orchestrated = Orchestrate(OpenAI, nodes=40) #handles all the task assignment and allocation and agent communication using a vectorstore as a universal communication layer and also handlles the task completion logic

Objective = "Make a business website for a marketing consultancy"

Swarms = (Swarms(orchestrated, auto=True, Objective))
```





In terms of architecture, the swarm might look something like this:

```
                                           (Orchestrator)
                                             /        \
            ( Agent)---(Communication Layer)       (Communication Layer)---(Agent)
              /                  |                                           |                 \
(Task Assignment)      (Task Completion)                    (Task Assignment)       (Task Completion)
```

Each LLM agent communicates with the orchestrator through a dedicated communication layer. The orchestrator assigns tasks to each LLM agent, which the agents then complete and return. This setup allows for a high degree of flexibility, scalability, and robustness.

In the context of swarm LLMs, one could consider an **Omni-Vector Embedding Database** for communication. This database could store and manage the high-dimensional vectors produced by each LLM agent.

- Strengths: This approach would allow for similarity-based lookup and matching of LLM-generated vectors, which can be particularly useful for tasks that involve finding similar outputs or recognizing patterns.

- Weaknesses: An Omni-Vector Embedding Database might add complexity to the system in terms of setup and maintenance. It might also require significant computational resources, depending on the volume of data being handled and the complexity of the vectors. The handling and transmission of high-dimensional vectors could also pose challenges in terms of network load.


from swarms import WorkerNode, Orchestrate

Orchestrate(WorkerNode, autoscale=True, nodes=int, swarm_type="flat")

"""
#in a shared state, provide 3 communication times, during task assignment, task compeltion, and feedback or inability to complete a task.
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from swarms.agents.memory.ocean import OceanDB


class Orchestrator(ABC):
    def __init__(self, 
    agent, 
    agent_list: List[Any], 
    task_queue: List[Any], 
    vector_db: OceanDB
    ):
        self.agent = agent
        self.agents = [agent_class() for _ in range(agent_list)]
        self.task_queue = task_queue
        self.vector_db = vector_db
        self.current_tasks = {}
        self.lock = threading.Lock()
        
    @abstractmethod
    def assign_task(self, agent_id: int, task: Dict[str, Any]) -> None:
        """Assign a task to a specific agent"""
        with self.lock:
            if self.task_queue:
                #get and agent and a task
                agent = self.agents.pop(0)
                task = self.task_queue.popleft()

                #process the task and get result and vector representation
                result, vector_representation = agent.process_task()

                #store the vector representation in the database
                self.vector_db.add_documents([vector_representation],[str(id(task))])

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
            results = self.vector_db.query(query_texts=[str(agent_id)], n_results=10)
            return results
        except Exception as e:
            logging.error(f"Failed to retrieve results from agent {agent_id}. Error {e}")
            raise
    
    @abstractmethod
    def update_vector_db(self, data) -> None:
        """Update the vector database"""
        try:
            self.vector_db.add_documents([data['vector']], [str(data['task_id'])])
        except Exception as e:
            logging.error(f"Failed to update the vector database. Error: {e}")
            raise


    @abstractmethod
    def get_vector_db(self):
        """Retrieve the vector database"""
        return self.vector_db

    def append_to_db(self, collection, result: str):
        """append the result of the swarm to a specifici collection in the database"""
        try:
            self.vector_db.append_document(collection, result, id=str(id(result)))
        except Exception as e:
            logging.error(f"Failed to append the agent output to database. Error: {e}")
            raise

    def run(self, objective:str, collection):
        """Runs"""

        if not objective or not isinstance(objective, str):
            logging.error("Invalid objective")
            raise ValueError("A valid objective is required")
        
        try:
            #add objective to agent
            self.task_queue.append(objective)

            #assign tasks to agents
            results = [self.assign_task(agent_id, task) for agent_id, task in zip(range(len(self.agents)), self.task_queue)]
            
            for result in results:
                self.append_to_db(collection, result)
            

            logging.info(f"Successfully ran swarms with results: {results}")
            return results
        except Exception as e:
            logging.error(f"An error occured in swarm: {e}")
            return None




#PRE CONFIGURED AGENTS WITH domain explicit TOOLS
#Build your own Agent
# Learn from previous runs in session management => it's a sucessful run => omniversal memory for all swarms  