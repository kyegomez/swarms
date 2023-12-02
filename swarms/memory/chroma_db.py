import os
from termcolor import colored
import logging
from typing import Dict, List, Optional
import chromadb
import tiktoken as tiktoken
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
from swarms.utils.token_count_tiktoken import limit_tokens_from_string

load_dotenv()

# ChromaDB settings
client = chromadb.Client(Settings(anonymized_telemetry=False))


# ChromaDB client
def get_chromadb_client():
    return client


#  OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Results storage using local ChromaDB
class ChromaDB:
    """

    ChromaDB database

    Args:
        metric (str): _description_
        RESULTS_STORE_NAME (str): _description_
        LLM_MODEL (str): _description_
        openai_api_key (str): _description_

    Methods:
        add: _description_
        query: _description_

    Examples:
        >>> chromadb = ChromaDB(
        >>>     metric="cosine",
        >>>     RESULTS_STORE_NAME="results",
        >>>     LLM_MODEL="gpt3",
        >>>     openai_api_key=OPENAI_API_KEY,
        >>> )
        >>> chromadb.add(task, result, result_id)
        >>> chromadb.query(query, top_results_num)
    """

    def __init__(
        self,
        metric: str,
        RESULTS_STORE_NAME: str,
        LLM_MODEL: str,
        openai_api_key: str = OPENAI_API_KEY,
        top_results_num: int = 3,
        limit_tokens: Optional[int] = 1000,
    ):
        self.metric = metric
        self.RESULTS_STORE_NAME = RESULTS_STORE_NAME
        self.LLM_MODEL = LLM_MODEL
        self.openai_api_key = openai_api_key
        self.top_results_num = top_results_num
        self.limit_tokens = limit_tokens

        # Disable ChromaDB logging
        logging.getLogger("chromadb").setLevel(logging.ERROR)
        # Create Chroma collection
        chroma_persist_dir = "chroma"
        chroma_client = chromadb.PersistentClient(
            settings=chromadb.config.Settings(
                persist_directory=chroma_persist_dir,
            )
        )

        # Create embedding function
        embedding_function = OpenAIEmbeddingFunction(
            api_key=openai_api_key
        )

        # Create Chroma collection
        self.collection = chroma_client.get_or_create_collection(
            name=RESULTS_STORE_NAME,
            metadata={"hnsw:space": metric},
            embedding_function=embedding_function,
        )

    def add(self, task: Dict, result: str, result_id: str):
        """Adds a result to the ChromaDB collection

        Args:
            task (Dict): _description_
            result (str): _description_
            result_id (str): _description_
        """

        try:
            # Embed the result
            embeddings = (
                self.collection.embedding_function.embed([result])[0]
                .tolist()
                .copy()
            )

            # If the result is a list, flatten it
            if (
                len(
                    self.collection.get(ids=[result_id], include=[])[
                        "ids"
                    ]
                )
                > 0
            ):  # Check if the result already exists
                self.collection.update(
                    ids=result_id,
                    embeddings=embeddings,
                    documents=result,
                    metadatas={
                        "task": task["task_name"],
                        "result": result,
                    },
                )

            # If the result is not a list, add it
            else:
                self.collection.add(
                    ids=result_id,
                    embeddings=embeddings,
                    documents=result,
                    metadatas={
                        "task": task["task_name"],
                        "result": result,
                    },
                )
        except Exception as error:
            print(
                colored(f"Error adding to ChromaDB: {error}", "red")
            )

    def query(
        self,
        query: str,
    ) -> List[dict]:
        """Queries the ChromaDB collection with a query for the top results

        Args:
            query (str): _description_
            top_results_num (int): _description_

        Returns:
            List[dict]: _description_
        """
        try:
            count: int = self.collection.count()
            if count == 0:
                return []
            results = self.collection.query(
                query_texts=query,
                n_results=min(self.top_results_num, count),
                include=["metadatas"],
            )
            out = [item["task"] for item in results["metadatas"][0]]
            out = limit_tokens_from_string(
                out, "gpt-4", self.limit_tokens
            )
            return out
        except Exception as error:
            print(colored(f"Error querying ChromaDB: {error}", "red"))
