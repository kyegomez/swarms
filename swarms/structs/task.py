from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Union,
)

from swarms.structs.agent import Agent


# Define a generic Task that can handle different types of callable objects
@dataclass
class Task:
    """
    Task class for running a task in a sequential workflow.


    Args:
        description (str): The description of the task.
        agent (Union[Callable, Agent]): The model or agent to execute the task.
        args (List[Any]): Additional arguments to pass to the task execution.
        kwargs (Dict[str, Any]): Additional keyword arguments to pass to the task execution.
        result (Any): The result of the task execution.
        history (List[Any]): The history of the task execution.

    Methods:
        execute: Execute the task.


    Examples:
    >>> from swarms.structs import Task, Agent
    >>> from swarms.models import OpenAIChat
    >>> agent = Agent(llm=OpenAIChat(openai_api_key=""), max_loops=1, dashboard=False)
    >>> task = Task(description="What's the weather in miami", agent=agent)
    >>> task.execute()
    >>> task.result

    """

    description: str
    agent: Union[Callable, Agent]
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    result: Any = None
    history: List[Any] = field(default_factory=list)
    # logger = logging.getLogger(__name__)

    def execute(self):
        """
        Execute the task.

        Raises:
            ValueError: If a Agent instance is used as a task and the 'task' argument is not provided.
        """
        if isinstance(self.agent, Agent):
            # Add a prompt to notify the Agent of the sequential workflow
            if "prompt" in self.kwargs:
                self.kwargs["prompt"] += (
                    f"\n\nPrevious output: {self.result}"
                    if self.result
                    else ""
                )
            else:
                self.kwargs["prompt"] = (
                    f"Main task: {self.description}"
                    + (
                        f"\n\nPrevious output: {self.result}"
                        if self.result
                        else ""
                    )
                )
            self.result = self.agent.run(*self.args, **self.kwargs)
        else:
            self.result = self.agent(*self.args, **self.kwargs)

        self.history.append(self.result)
