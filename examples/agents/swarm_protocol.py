from dataclasses import dataclass
from typing import List

from swarms import JSON, BaseLLM, BaseVectorDatabase, Agent


@dataclass
class YourAgent(Agent):
    """
    Represents an agent in the swarm protocol.

    Attributes:
        llm (BaseLLM): The low-level module for the agent.
        long_term_memory (BaseVectorDatabase): The long-term memory for the agent.
        tool_schema (List[JSON]): The schema for the tools used by the agent.
    """

    llm: BaseLLM
    long_term_memory: BaseVectorDatabase
    tool_schema: JSON
    tool_schemas: List[JSON]

    def step(self, task: str, *args, **kwargs):
        """
        Performs a single step in the agent's task.

        Args:
            task (str): The task to be performed.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        ...

    def run(self, task: str, *args, **kwargs):
        """
        Runs the agent's task.

        Args:
            task (str): The task to be performed.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        ...

    def plan(self, task: str, *args, **kwargs):
        """
        Plans the agent's task.

        Args:
            task (str): The task to be performed.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        ...
