import asyncio
import inspect
from typing import Any, Dict, List, Optional, Union

from swarms.structs.agent import Agent
from swarms.structs.context_utils import (
    agent_answer,
    messages_for,
    run_on_conversation,
    split_last_turn,
)
from swarms.structs.conversation import Conversation
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.loguru_logger import initialize_logger
from swarms.utils.output_types import OutputType

logger = initialize_logger(log_folder="one_to_one")


async def _a_run_agent_on_conversation(
    agent: Agent,
    conversation: Conversation,
    **kwargs,
) -> str:
    """Run an agent asynchronously on the shared conversation and record its answer."""
    agent_name = getattr(agent, "agent_name", "Agent")
    prior, task = split_last_turn(
        messages_for(agent_name, conversation)
    )
    if hasattr(agent, "arun") and callable(agent.arun):
        try:
            response = await agent.arun(task=task, messages=prior, **kwargs)
        except TypeError:
            try:
                response = await agent.arun(task, **kwargs)
            except TypeError:
                response = await agent.arun(task)
    elif hasattr(agent, "run") and callable(agent.run):
        try:
            response = await asyncio.to_thread(
                agent.run, task=task, messages=prior, **kwargs
            )
        except TypeError:
            try:
                response = await asyncio.to_thread(agent.run, task, **kwargs)
            except TypeError:
                response = await asyncio.to_thread(agent.run, task)
    elif callable(agent):
        if inspect.iscoroutinefunction(agent):
            response = await agent(task)
        else:
            response = await asyncio.to_thread(agent, task)
    else:
        raise TypeError(
            f"Agent {agent} is neither callable nor has run/arun method."
        )

    answer = agent_answer(agent, fallback=response)
    conversation.add(role=agent_name, content=answer)
    return answer


def one_to_one(
    sender: Agent,
    receiver: Agent,
    task: str,
    max_loops: int = 1,
    output_type: OutputType = "dict",
) -> Union[Dict[str, Any], List[str], str]:
    """
    One-to-one communication between two agents.

    The sender answers the task, the receiver answers the sender, and the
    exchange repeats ``max_loops`` times. Each agent reads the shared
    conversation as typed turns, so it can tell its own earlier messages
    from the other agent's.

    Args:
        sender (Agent): The agent that speaks first.
        receiver (Agent): The agent that replies.
        task (str): The task to be processed.
        max_loops (int, optional): Number of sender/receiver exchanges.
            Defaults to 1.
        output_type (OutputType, optional): The format of the output.
            Defaults to "dict".

    Returns:
        Union[Dict[str, Any], List[str], str]: The conversation history in
            the requested format.

    Raises:
        ValueError: If sender, receiver, or task is empty.
    """
    if not sender or not receiver or not task:
        raise ValueError(
            "Sender, receiver, and task cannot be empty."
        )

    conversation = Conversation()
    conversation.add(role="User", content=task)

    try:
        for _ in range(max_loops):
            run_on_conversation(sender, conversation)
            run_on_conversation(receiver, conversation)
    except Exception as error:
        logger.error(
            f"Error during one_to_one communication: {error}"
        )
        raise error

    return history_output_formatter(conversation, output_type)


async def a_one_to_one(
    sender: Agent,
    receiver: Agent,
    task: str,
    max_loops: int = 1,
    output_type: OutputType = "dict",
    *args,
    **kwargs,
) -> Union[Dict[str, Any], List[str], str]:
    """
    Asynchronously execute one-to-one communication between two agents on the event loop.

    The sender answers the task, the receiver answers the sender, and the
    exchange repeats max_loops times.

    Args:
        sender (Agent): The agent that speaks first.
        receiver (Agent): The agent that replies.
        task (str): The task to be processed.
        max_loops (int, optional): Number of sender/receiver exchanges. Defaults to 1.
        output_type (OutputType, optional): Format of the output. Defaults to "dict".

    Returns:
        Union[Dict[str, Any], List[str], str]: Conversation history in requested format.

    Raises:
        ValueError: If sender, receiver, or task is empty.
    """
    if not sender or not receiver or not task:
        raise ValueError(
            "Sender, receiver, and task cannot be empty."
        )

    conversation = Conversation()
    conversation.add(role="User", content=task)

    try:
        for _ in range(max_loops):
            await _a_run_agent_on_conversation(sender, conversation, **kwargs)
            await _a_run_agent_on_conversation(receiver, conversation, **kwargs)
    except Exception as error:
        logger.error(
            f"Error during async one_to_one communication: {error}"
        )
        raise error

    return history_output_formatter(conversation, output_type)


class OneToOne:
    """
    Facilitates one-to-one communication between two agents.

    A reusable wrapper around one-to-one communication that holds the two agents
    and the output format. Supports both synchronous run and native async arun / batch_arun
    on a single event loop.
    """

    def __init__(
        self,
        sender: Agent,
        receiver: Agent,
        name: str = "OneToOne",
        description: str = "A one-to-one communication pattern between two agents",
        output_type: OutputType = "dict",
    ):
        """
        Initialize the OneToOne communication.

        Args:
            sender: The sender agent
            receiver: The receiver agent
            name: Name of the communication pattern
            description: Description of the communication pattern's purpose
            output_type: Type of output format, one of 'dict', 'list', 'string', 'json', 'yaml', etc.
        """
        self.sender = sender
        self.receiver = receiver
        self.name = name
        self.description = description
        self.output_type = output_type

    def run(
        self, task: str, max_loops: int = 1
    ) -> Union[Dict[str, Any], List[str], str]:
        """
        Run the one-to-one communication with the given task.

        Args:
            task: Task to be processed
            max_loops: Number of exchange iterations

        Returns:
            Union[Dict, List, str]: The conversation history in the requested format
        """
        return one_to_one(
            sender=self.sender,
            receiver=self.receiver,
            task=task,
            max_loops=max_loops,
            output_type=self.output_type,
        )

    async def arun(
        self,
        task: str,
        max_loops: Optional[int] = None,
        *args,
        **kwargs,
    ) -> Union[Dict[str, Any], List[str], str]:
        """
        Asynchronously run the one-to-one communication on the event loop.

        Args:
            task: Task to be processed
            max_loops: Optional number of iterations (defaults to 1)

        Returns:
            Union[Dict, List, str]: The conversation history in requested format
        """
        loops = max_loops if max_loops is not None else 1
        return await a_one_to_one(
            sender=self.sender,
            receiver=self.receiver,
            task=task,
            max_loops=loops,
            output_type=self.output_type,
            *args,
            **kwargs,
        )

    async def batch_arun(
        self,
        tasks: List[str],
        max_loops: Optional[int] = None,
        *args,
        **kwargs,
    ) -> List[Union[Dict[str, Any], List[str], str]]:
        """
        Asynchronously run multiple one-to-one dialogues concurrently.

        Args:
            tasks: List of task strings to process
            max_loops: Optional number of iterations

        Returns:
            List of conversation histories in requested format
        """
        loops = max_loops if max_loops is not None else 1
        coros = [
            self.arun(task=task, max_loops=loops, *args, **kwargs)
            for task in tasks
        ]
        return await asyncio.gather(*coros)
