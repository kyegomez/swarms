import os
from typing import List

import faiss
from langchain_community.docstore import InMemoryDocstore
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_experimental.autonomous_agents import AutoGPT

from swarms.tools.tool import BaseTool
from swarms.utils.decorators import error_decorator, timing_decorator


class Worker:
    """
    The Worker class represents an autonomous agent that can perform tassks through
    function calls or by running a chat.

    Args:
        name (str, optional): Name of the agent. Defaults to "Autobot Swarm Worker".
        role (str, optional): Role of the agent. Defaults to "Worker in a swarm".
        external_tools (list, optional): List of external tools. Defaults to None.
        human_in_the_loop (bool, optional): Whether to include human in the loop. Defaults to False.
        temperature (float, optional): Temperature for the agent. Defaults to 0.5.
        llm ([type], optional): Language model. Defaults to None.
        openai_api_key (str, optional): OpenAI API key. Defaults to None.

    Raises:
        RuntimeError: If there is an error while setting up the agent.

    Example:
    >>> worker = Worker(
    ...     name="My Worker",
    ...     role="Worker",
    ...     external_tools=[MyTool1(), MyTool2()],
    ...     human_in_the_loop=False,
    ...     temperature=0.5,
    ... )
    >>> worker.run("What's the weather in Miami?")

    """

    def __init__(
        self,
        name: str = "WorkerAgent",
        role: str = "Worker in a swarm",
        external_tools=None,
        human_in_the_loop: bool = False,
        temperature: float = 0.5,
        llm=None,
        openai_api_key: str = None,
        tools: List[BaseTool] = None,
        embedding_size: int = 1536,
        search_kwargs: dict = {"k": 8},
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        self.name = name
        self.role = role
        self.external_tools = external_tools
        self.human_in_the_loop = human_in_the_loop
        self.temperature = temperature
        self.llm = llm
        self.openai_api_key = openai_api_key
        self.tools = tools
        self.embedding_size = embedding_size
        self.search_kwargs = search_kwargs
        self.verbose = verbose

        self.setup_tools(external_tools)
        self.setup_memory()
        self.setup_agent()

    def reset(self):
        """
        Reset the message history.
        """
        self.message_history = []

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
                name="My Worker",
                role="Worker",
                external_tools=external_tools,
                human_in_the_loop=False,
                temperature=0.5)
        ```
        """
        if self.tools is None:
            self.tools = []

        if external_tools is not None:
            self.tools.extend(external_tools)

    def setup_memory(self):
        """
        Set up memory for the worker.
        """
        openai_api_key = (
            os.getenv("OPENAI_API_KEY") or self.openai_api_key
        )
        try:
            embeddings_model = OpenAIEmbeddings(
                openai_api_key=openai_api_key
            )
            embedding_size = self.embedding_size
            index = faiss.IndexFlatL2(embedding_size)

            self.vectorstore = FAISS(
                embeddings_model.embed_query,
                index,
                InMemoryDocstore({}),
                {},
            )

        except Exception as error:
            raise RuntimeError(
                "Error setting up memory perhaps try try tuning the"
                f" embedding size: {error}"
            )

    def setup_agent(self):
        """
        Set up the autonomous agent.
        """
        try:
            self.agent = AutoGPT.from_llm_and_tools(
                ai_name=self.name,
                ai_role=self.role,
                tools=self.tools,
                llm=self.llm,
                memory=self.vectorstore.as_retriever(
                    search_kwargs=self.search_kwargs
                ),
                human_in_the_loop=self.human_in_the_loop,
            )

        except Exception as error:
            raise RuntimeError(f"Error setting up agent: {error}")

    # @log_decorator
    @error_decorator
    @timing_decorator
    def run(self, task: str = None, *args, **kwargs):
        """
        Run the autonomous agent on a given task.

        Parameters:
        - `task`: The task to be processed.

        Returns:
        - `result`: The result of the agent's processing.
        """
        try:
            result = self.agent.run([task], *args, **kwargs)
            return result
        except Exception as error:
            raise RuntimeError(f"Error while running agent: {error}")

    # @log_decorator
    @error_decorator
    @timing_decorator
    def __call__(self, task: str = None, *args, **kwargs):
        """
        Make the worker callable to run the agent on a given task.

        Parameters:
        - `task`: The task to be processed.

        Returns:
        - `results`: The results of the agent's processing.
        """
        try:
            result = self.agent.run([task], *args, **kwargs)
            return result
        except Exception as error:
            raise RuntimeError(f"Error while running agent: {error}")
