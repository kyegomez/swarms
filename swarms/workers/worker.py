import os
import random
from typing import Dict, Union

import faiss
from langchain.chains.qa_with_sources.loading import (
    load_qa_with_sources_chain,)
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.tools import ReadFileTool, WriteFileTool
from langchain.tools.human.tool import HumanInputRun
from langchain.vectorstores import FAISS
from langchain_experimental.autonomous_agents import AutoGPT

from swarms.agents.message import Message
from swarms.tools.autogpt import (
    WebpageQATool,
    process_csv,
)
from swarms.utils.decorators import error_decorator, timing_decorator

# cache
ROOT_DIR = "./data/"


class Worker:
    """
    Useful for when you need to spawn an autonomous agent instance as a worker to accomplish complex tasks,
    it can search the internet or spawn child multi-modality models to process and generate images and text or audio and so on

    Parameters:
    - `model_name` (str): The name of the language model to be used (default: "gpt-4").
    - `openai_api_key` (str): The OpenvAI API key (optional).
    - `ai_name` (str): The name of the AI worker.
    - `ai_role` (str): The role of the AI worker.
    - `external_tools` (list): List of external tools (optional).
    - `human_in_the_loop` (bool): Enable human-in-the-loop interaction (default: False).
    - `temperature` (float): The temperature parameter for response generation (default: 0.5).
    - `llm` (ChatOpenAI): Pre-initialized ChatOpenAI model instance (optional).
    - `openai` (bool): If True, use the OpenAI language model; otherwise, use `llm` (default: True).

    Usage
    ```
    from swarms import Worker

    node = Worker(
        ai_name="Optimus Prime",

    )

    task = "What were the winning boston marathon times for the past 5 years (ending in 2022)? Generate a table of the year, name, country of origin, and times."
    response = node.run(task)
    print(response)
    ```

    llm + tools + memory

    """

    def __init__(
        self,
        ai_name: str = "Autobot Swarm Worker",
        ai_role: str = "Worker in a swarm",
        external_tools=None,
        human_in_the_loop=False,
        temperature: float = 0.5,
        llm=None,
        openai_api_key: str = None,
    ):
        self.temperature = temperature
        self.human_in_the_loop = human_in_the_loop
        self.llm = llm
        self.openai_api_key = openai_api_key
        self.ai_name = ai_name
        self.ai_role = ai_role
        self.coordinates = (
            random.randint(0, 100),
            random.randint(0, 100),
        )  # example coordinates for proximity

        self.setup_tools(external_tools)
        self.setup_memory()
        self.setup_agent()

    def reset(self):
        """
        Reset the message history.
        """
        self.message_history = ["Here is the conversation so far"]

    @property
    def name(self):
        """Name of the agent"""
        return self.ai_name

    def receieve(self, name: str, message: str) -> None:
        """
        Receive a message and update the message history.

        Parameters:
        - `name` (str): The name of the sender.
        - `message` (str): The received message.
        """
        self.message_history.append(f"{name}: {message}")

    def send(self) -> str:
        """Send message history."""
        self.agent.run(task=self.message_history)

    def add(self, task, priority=0):
        """Add a task to the task queue."""
        self.task_queue.append((priority, task))

    def setup_tools(self, external_tools):
        """
        Set up tools for the worker.

        Parameters:
        - `external_tools` (list): List of external tools (optional).

        Example:
        ```
        external_tools = [MyTool1(), MyTool2()]
        worker = Worker(model_name="gpt-4",
                openai_api_key="my_key",
                ai_name="My Worker",
                ai_role="Worker",
                external_tools=external_tools,
                human_in_the_loop=False,
                temperature=0.5)
        ```
        """
        query_website_tool = WebpageQATool(
            qa_chain=load_qa_with_sources_chain(self.llm))

        self.tools = [
            WriteFileTool(root_dir=ROOT_DIR),
            ReadFileTool(root_dir=ROOT_DIR),
            process_csv,
            query_website_tool,
            HumanInputRun(),
            # compile,
            # VQAinference,
        ]
        if external_tools is not None:
            self.tools.extend(external_tools)

    def setup_memory(self):
        """
        Set up memory for the worker.
        """
        openai_api_key = os.getenv("OPENAI_API_KEY") or self.openai_api_key
        try:
            embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)
            embedding_size = 1536
            index = faiss.IndexFlatL2(embedding_size)

            self.vectorstore = FAISS(embeddings_model.embed_query, index,
                                     InMemoryDocstore({}), {})

        except Exception as error:
            raise RuntimeError(
                "Error setting up memory perhaps try try tuning the embedding size:"
                f" {error}")

    def setup_agent(self):
        """
        Set up the autonomous agent.
        """
        try:
            self.agent = AutoGPT.from_llm_and_tools(
                ai_name=self.ai_name,
                ai_role=self.ai_role,
                tools=self.tools,
                llm=self.llm,
                memory=self.vectorstore.as_retriever(search_kwargs={"k": 8}),
                human_in_the_loop=self.human_in_the_loop,
            )

        except Exception as error:
            raise RuntimeError(f"Error setting up agent: {error}")

    # @log_decorator
    @error_decorator
    @timing_decorator
    def run(self, task: str = None):
        """
        Run the autonomous agent on a given task.

        Parameters:
        - `task`: The task to be processed.

        Returns:
        - `result`: The result of the agent's processing.
        """
        try:
            result = self.agent.run([task])
            return result
        except Exception as error:
            raise RuntimeError(f"Error while running agent: {error}")

    # @log_decorator
    @error_decorator
    @timing_decorator
    def __call__(self, task: str = None):
        """
        Make the worker callable to run the agent on a given task.

        Parameters:
        - `task`: The task to be processed.

        Returns:
        - `results`: The results of the agent's processing.
        """
        try:
            results = self.agent.run([task])
            return results
        except Exception as error:
            raise RuntimeError(f"Error while running agent: {error}")

    def health_check(self):
        pass

    # @log_decorator
    @error_decorator
    @timing_decorator
    def chat(self, msg: str = None, streaming: bool = False):
        """
        Run chat

        Args:
            msg (str, optional): Message to send to the agent. Defaults to None.
            language (str, optional): Language to use. Defaults to None.
            streaming (bool, optional): Whether to stream the response. Defaults to False.

        Returns:
            str: Response from the agent

        Usage:
        --------------
        agent = MultiModalAgent()
        agent.chat("Hello")

        """

        # add users message to the history
        self.history.append(Message("User", msg))

        # process msg
        try:
            response = self.agent.run(msg)

            # add agent's response to the history
            self.history.append(Message("Agent", response))

            # if streaming is = True
            if streaming:
                return self._stream_response(response)
            else:
                response

        except Exception as error:
            error_message = f"Error processing message: {str(error)}"

            # add error to history
            self.history.append(Message("Agent", error_message))

            return error_message

    def _stream_response(self, response: str = None):
        """
        Yield the response token by token (word by word)

        Usage:
        --------------
        for token in _stream_response(response):
            print(token)

        """
        for token in response.split():
            yield token

    @staticmethod
    def _message_to_dict(message: Union[Dict, str]):
        """Convert a message"""
        if isinstance(message, str):
            return {"content": message}
        else:
            return message

    def is_within_proximity(self, other_worker):
        """Using Euclidean distance for proximity check"""
        distance = ((self.coordinates[0] - other_worker.coordinates[0])**2 +
                    (self.coordinates[1] - other_worker.coordinates[1])**2)**0.5
        return distance < 10  # threshold for proximity
