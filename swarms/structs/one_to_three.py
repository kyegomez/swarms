import asyncio
import inspect
from typing import Any, Dict, List, Optional, Tuple, Union

from swarms.structs.agent import Agent
from swarms.structs.context_utils import (
    agent_answer,
    messages_for,
    run_on_conversation,
    split_last_turn,
)
from swarms.structs.conversation import Conversation
from swarms.structs.omni_agent_types import AgentListType
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.loguru_logger import initialize_logger
from swarms.utils.output_types import OutputType

logger = initialize_logger(log_folder="one_to_three")


def _check_receivers(receivers: AgentListType) -> None:
    if receivers is None or len(receivers) != 3:
        raise ValueError("The number of receivers must be exactly 3.")


async def _a_run_agent(
    agent: Agent,
    task: str,
    messages: Optional[List[Dict[str, str]]] = None,
    **kwargs,
) -> Any:
    """Run an agent asynchronously, preferring native arun with fallback to thread pool."""
    if hasattr(agent, "arun") and callable(agent.arun):
        try:
            return await agent.arun(task=task, messages=messages, **kwargs)
        except TypeError:
            try:
                return await agent.arun(task, **kwargs)
            except TypeError:
                return await agent.arun(task)
    elif hasattr(agent, "run") and callable(agent.run):
        try:
            return await asyncio.to_thread(
                agent.run, task=task, messages=messages, **kwargs
            )
        except TypeError:
            try:
                return await asyncio.to_thread(agent.run, task, **kwargs)
            except TypeError:
                return await asyncio.to_thread(agent.run, task)
    elif callable(agent):
        if inspect.iscoroutinefunction(agent):
            return await agent(task)
        else:
            return await asyncio.to_thread(agent, task)
    else:
        raise TypeError(
            f"Agent {agent} is neither callable nor has run/arun method."
        )


async def _execute_receiver_async(
    agent: Agent,
    conversation: Conversation,
    semaphore: Optional[asyncio.Semaphore] = None,
    **kwargs,
) -> Tuple[str, str]:
    """Execute a single receiver agent against the broadcast conversation."""
    agent_name = getattr(agent, "agent_name", "Receiver")
    prior, turn_task = split_last_turn(
        messages_for(agent_name, conversation)
    )
    if semaphore:
        async with semaphore:
            resp = await _a_run_agent(
                agent, task=turn_task, messages=prior, **kwargs
            )
    else:
        resp = await _a_run_agent(
            agent, task=turn_task, messages=prior, **kwargs
        )
    answer = agent_answer(agent, fallback=resp)
    return agent_name, answer


def one_to_three(
    sender: Agent,
    receivers: AgentListType,
    task: str,
    output_type: OutputType = "dict",
) -> Union[Dict[str, Any], List[str], str]:
    """
    One-to-three communication from one agent to exactly three agents.

    The sender answers the task, then each of the three receivers reads the
    shared conversation and replies to it in turn.

    Args:
        sender (Agent): The agent sending the message.
        receivers (AgentListType): Exactly three receiving agents.
        task (str): The task to be processed.
        output_type (OutputType, optional): The format of the output.
            Defaults to "dict".

    Returns:
        Union[Dict[str, Any], List[str], str]: The conversation history in
            the requested format.

    Raises:
        ValueError: If sender or task is empty, or there are not exactly
            three receivers.
    """
    _check_receivers(receivers)
    if not sender or not task:
        raise ValueError("Sender and task cannot be empty.")

    conversation = Conversation()
    conversation.add(role="User", content=task)

    try:
        run_on_conversation(sender, conversation)

        for agent in receivers:
            run_on_conversation(agent, conversation)
    except Exception as error:
        logger.error(f"Error in one_to_three: {error}")
        raise error

    return history_output_formatter(conversation, output_type)


async def a_one_to_three(
    sender: Agent,
    receivers: AgentListType,
    task: str,
    output_type: OutputType = "dict",
    max_concurrency: Optional[int] = None,
    *args,
    **kwargs,
) -> Union[Dict[str, Any], List[str], str]:
    """
    Asynchronously broadcast communication from one agent to exactly three agents.

    The sender answers the task first on the event loop, then the three receivers
    fan out concurrently via native async execution.

    Args:
        sender (Agent): The agent sending the message.
        receivers (AgentListType): Exactly three receiving agents.
        task (str): The task to be processed.
        output_type (OutputType, optional): Format of the output. Defaults to "dict".
        max_concurrency (Optional[int], optional): Maximum concurrent receiver agents.

    Returns:
        Union[Dict[str, Any], List[str], str]: Conversation history in requested format.

    Raises:
        ValueError: If sender or task is empty, or there are not exactly three receivers.
    """
    _check_receivers(receivers)
    if not sender or not task:
        raise ValueError("Sender and task cannot be empty.")

    conversation = Conversation()
    conversation.add(role="User", content=task)

    try:
        # Sender processes initial prompt
        sender_name = getattr(sender, "agent_name", "Sender")
        sender_prior, sender_task = split_last_turn(
            messages_for(sender_name, conversation)
        )
        sender_resp = await _a_run_agent(
            sender, task=sender_task, messages=sender_prior, **kwargs
        )
        sender_answer = agent_answer(sender, fallback=sender_resp)
        conversation.add(role=sender_name, content=sender_answer)

        # Receivers fan out concurrently on event loop
        semaphore = (
            asyncio.Semaphore(max_concurrency)
            if max_concurrency and max_concurrency > 0
            else None
        )
        tasks = [
            _execute_receiver_async(
                agent=rcv,
                conversation=conversation,
                semaphore=semaphore,
                **kwargs,
            )
            for rcv in receivers
        ]
        receiver_results = await asyncio.gather(*tasks)

        for agent_name, answer in receiver_results:
            conversation.add(role=agent_name, content=answer)

    except Exception as error:
        logger.error(f"Error in async one_to_three: {error}")
        raise error

    return history_output_formatter(conversation, output_type)


class OneToThree:
    """
    Facilitates one-to-three communication from one agent to exactly three agents.

    A reusable wrapper around one-to-three communication that holds the sender,
    the three receivers and the output format. Supports both synchronous run and
    native async arun / batch_arun on a single event loop.
    """

    def __init__(
        self,
        sender: Agent,
        receivers: AgentListType,
        name: str = "OneToThree",
        description: str = "A one-to-three communication pattern from one agent to exactly three agents",
        output_type: OutputType = "dict",
        max_concurrency: Optional[int] = None,
    ):
        """
        Initialize the OneToThree communication.

        Args:
            sender: The sender agent
            receivers: List of exactly three receiver agents
            name: Name of the communication pattern
            description: Description of the communication pattern's purpose
            output_type: Type of output format, one of 'dict', 'list', 'string', 'json', 'yaml', etc.
            max_concurrency: Optional limit on concurrently running receiver agents in arun.

        Raises:
            ValueError: If there are not exactly three receivers.
        """
        _check_receivers(receivers)

        self.sender = sender
        self.receivers = receivers
        self.name = name
        self.description = description
        self.output_type = output_type
        self.max_concurrency = max_concurrency

    def run(self, task: str) -> Union[Dict[str, Any], List[str], str]:
        """
        Run the one-to-three communication with the given task.

        Args:
            task: Task to be processed

        Returns:
            Union[Dict, List, str]: The conversation history in the requested format
        """
        return one_to_three(
            sender=self.sender,
            receivers=self.receivers,
            task=task,
            output_type=self.output_type,
        )

    async def arun(
        self,
        task: str,
        max_concurrency: Optional[int] = None,
        *args,
        **kwargs,
    ) -> Union[Dict[str, Any], List[str], str]:
        """
        Asynchronously run the one-to-three communication on the event loop.

        Args:
            task: Task to be processed
            max_concurrency: Optional override for receiver concurrency limit

        Returns:
            Union[Dict, List, str]: The conversation history in the requested format
        """
        effective_concurrency = (
            max_concurrency
            if max_concurrency is not None
            else self.max_concurrency
        )
        return await a_one_to_three(
            sender=self.sender,
            receivers=self.receivers,
            task=task,
            output_type=self.output_type,
            max_concurrency=effective_concurrency,
            *args,
            **kwargs,
        )

    async def batch_arun(
        self,
        tasks: List[str],
        max_concurrency: Optional[int] = None,
        *args,
        **kwargs,
    ) -> List[Union[Dict[str, Any], List[str], str]]:
        """
        Asynchronously run multiple one-to-three broadcast tasks concurrently.

        Args:
            tasks: List of task strings to process
            max_concurrency: Optional override for receiver concurrency limit

        Returns:
            List of conversation histories in the requested format
        """
        effective_concurrency = (
            max_concurrency
            if max_concurrency is not None
            else self.max_concurrency
        )
        coros = [
            self.arun(
                task=task,
                max_concurrency=effective_concurrency,
                *args,
                **kwargs,
            )
            for task in tasks
        ]
        return await asyncio.gather(*coros)
