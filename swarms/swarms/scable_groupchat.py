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
    This is a class to enable scalable groupchat like a telegram, it takes an Worker as an input
    and handles all the logic to enable multi-agent collaboration at massive scale.

    Worker -> ScalableGroupChat(Worker * 10)
    -> every response is embedded and placed in chroma
    -> every response is then retrieved by querying the database and sent then passed into the prompt of the worker
    -> every worker is then updated with the new response
    -> every worker can communicate at any time
    -> every worker can communicate without restrictions in parallel

    """

    def __init__(
        self,
        worker_count: int = 5,
        collection_name: str = "swarm",
        api_key: str = None,
    ):
        self.workers = []
        self.worker_count = worker_count
        self.collection_name = collection_name
        self.api_key = api_key

        # Create a list of Worker instances with unique names
        for i in range(worker_count):
            self.workers.append(Worker(openai_api_key=api_key, ai_name=f"Worker-{i}"))

    def embed(self, input, model_name):
        """Embeds an input of size N into a vector of size M"""
        openai = embedding_functions.OpenAIEmbeddingFunction(
            api_key=self.api_key, model_name=model_name
        )

        embedding = openai(input)

        return embedding

    def retrieve_results(self, agent_id: int) -> Any:
        """Retrieve results from a specific agent"""

        try:
            # Query the vector database for documents created by the agents
            results = self.collection.query(query_texts=[str(agent_id)], n_results=10)

            return results
        except Exception as e:
            logging.error(
                f"Failed to retrieve results from agent {agent_id}. Error {e}"
            )
            raise

    # @abstractmethod
    def update_vector_db(self, data) -> None:
        """Update the vector database"""

        try:
            self.collection.add(
                embeddings=[data["vector"]],
                documents=[str(data["task_id"])],
                ids=[str(data["task_id"])],
            )

        except Exception as e:
            logging.error(f"Failed to update the vector database. Error: {e}")
            raise

    # @abstractmethod

    def get_vector_db(self):
        """Retrieve the vector database"""
        return self.collection

    def append_to_db(self, result: str):
        """append the result of the swarm to a specifici collection in the database"""

        try:
            self.collection.add(documents=[result], ids=[str(id(result))])

        except Exception as e:
            logging.error(f"Failed to append the agent output to database. Error: {e}")
            raise

    def chat(self, sender_id: int, receiver_id: int, message: str):
        """

        Allows the agents to chat with eachother thrught the vectordatabase

        # Instantiate the ScalableGroupChat with 10 agents
        orchestrator = ScalableGroupChat(
            llm,
            agent_list=[llm]*10,
            task_queue=[]
        )

        # Agent 1 sends a message to Agent 2
        orchestrator.chat(sender_id=1, receiver_id=2, message="Hello, Agent 2!")

        """
        if (
            sender_id < 0
            or sender_id >= self.worker_count
            or receiver_id < 0
            or receiver_id >= self.worker_count
        ):
            raise ValueError("Invalid sender or receiver ID")

        message_vector = self.embed(
            message,
        )

        # store the mesage in the vector database
        self.collection.add(
            embeddings=[message_vector],
            documents=[message],
            ids=[f"{sender_id}_to_{receiver_id}"],
        )

        self.run(objective=f"chat with agent {receiver_id} about {message}")
