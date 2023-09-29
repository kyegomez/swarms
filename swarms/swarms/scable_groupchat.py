import logging
from enum import Enum
from typing import Any

from chromadb.utils import embedding_functions

from swarms.workers.worker import Worker


class TaskStatus(Enum):
    QUEUED = 1
    RUNNING = 2
    COMPLETED = 3
    FAILED = 4

class ScalableGroupChat:
    """
    The Orchestrator takes in an agent, worker, or boss as input 
    then handles all the logic for
    - task creation,
    - task assignment,
    - and task compeletion.

    And, the communication for millions of agents to chat with eachother through
    a vector database that each agent has access to chat with.

    Each LLM agent chats with the orchestrator through a dedicated 
    communication layer. The orchestrator assigns tasks to each LLM agent,
    which the agents then complete and return. 

    This setup allows for a high degree of flexibility, scalability, and robustness.

    In the context of swarm LLMs, one could consider an **Omni-Vector Embedding Database
    for communication. This database could store and manage 
    the high-dimensional vectors produced by each LLM agent.

    Strengths: This approach would allow for similarity-based lookup and matching of 
    LLM-generated vectors, which can be particularly useful for tasks that involve finding similar outputs or recognizing patterns.

    Weaknesses: An Omni-Vector Embedding Database might add complexity to the system in terms of setup and maintenance. 
    It might also require significant computational resources, 
    depending on the volume of data being handled and the complexity of the vectors. 
    The handling and transmission of high-dimensional vectors could also pose challenges 
    in terms of network load.

    # Orchestrator
    * Takes in an agent class with vector store, 
    then handles all the communication and scales 
    up a swarm with number of agents and handles task assignment and task completion

    from swarms import OpenAI, Orchestrator, Swarm

    orchestrated = Orchestrate(OpenAI, nodes=40) #handles all the task assignment and allocation and agent communication using a vectorstore as a universal communication layer and also handlles the task completion logic

    Objective = "Make a business website for a marketing consultancy"

    Swarms = Swarms(orchestrated, auto=True, Objective))
    ```

    In terms of architecture, the swarm might look something like this:

    ```
                                            (Orchestrator)
                                                /        \
            Tools + Vector DB -- (LLM Agent)---(Communication Layer)       (Communication Layer)---(LLM Agent)-- Tools + Vector DB 
                /                  |                                           |                 \
    (Task Assignment)      (Task Completion)                    (Task Assignment)       (Task Completion)

    
    ###Usage
    ```
    from swarms import Orchestrator 
    
    # Instantiate the Orchestrator with 10 agents
    orchestrator = Orchestrator(llm, agent_list=[llm]*10, task_queue=[])

    # Add tasks to the Orchestrator
    tasks = [{"content": f"Write a short story about a {animal}."} for animal in ["cat", "dog", "bird", "fish", "lion", "tiger", "elephant", "giraffe", "monkey", "zebra"]]
    orchestrator.assign_tasks(tasks)

    # Run the Orchestrator
    orchestrator.run()

    # Retrieve the results
    for task in tasks:
    print(orchestrator.retrieve_result(id(task)))
    ```
    """
    def __init__(
        self,
        worker_count: int = 5,
        collection_name: str = "swarm",
        api_key: str = None,
        model_name: str = None,
        worker = None
    ):
        self.workers = []
        self.worker_count = worker_count

        # Create a list of Worker instances with unique names
        for i in range(worker_count):
            self.workers.append(Worker(openai_api_key="", ai_name=f"Worker-{i}"))
        
    def embed(self, input, api_key, model_name):
        openai = embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=model_name
        )
        embedding = openai(input)
        return embedding
                
    
    # @abstractmethod
    def retrieve_results(
        self, 
        agent_id: int
    ) -> Any:
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
    
    # @abstractmethod
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


    # @abstractmethod
    def get_vector_db(self):
        """Retrieve the vector database"""
        return self.collection
    

    def append_to_db(
        self, 
        result: str
    ):
        """append the result of the swarm to a specifici collection in the database"""

        try:
            self.collection.add(
                documents=[result],
                ids=[str(id(result))]
            )

        except Exception as e:
            logging.error(f"Failed to append the agent output to database. Error: {e}")
            raise

    
    
    def chat(
        self,
        sender_id: int,
        receiver_id: int,
        message: str
    ):
        """
        
        Allows the agents to chat with eachother thrught the vectordatabase

        # Instantiate the Orchestrator with 10 agents
        orchestrator = Orchestrator(
            llm, 
            agent_list=[llm]*10, 
            task_queue=[]
        )

        # Agent 1 sends a message to Agent 2
        orchestrator.chat(sender_id=1, receiver_id=2, message="Hello, Agent 2!")
            
        """
        if sender_id < 0 or sender_id >= self.worker_count or receiver_id < 0 or receiver_id >= self.worker_count:
            raise ValueError("Invalid sender or receiver ID")
        
        message_vector = self.embed(
            message,
            self.api_key,
            self.model_name
        )

        #store the mesage in the vector database
        self.collection.add(
            embeddings=[message_vector],
            documents=[message],
            ids=[f"{sender_id}_to_{receiver_id}"]
        )

        self.run(
            objective=f"chat with agent {receiver_id} about {message}"
        )


