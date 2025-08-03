from typing import Callable, Union

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)


def one_on_one_debate(
    max_loops: int = 1,
    task: str = None,
    agents: list[Union[Agent, Callable]] = None,
    img: str = None,
    output_type: str = "str-all-except-first",
) -> list:
    """
    Simulate a turn-based debate between two agents for a specified number of loops.

    Each agent alternately responds to the previous message, with the conversation
    history being tracked and available for both agents to reference. The debate
    starts with the provided `task` as the initial message.

    Args:
        max_loops (int): The number of conversational turns (each agent speaks per loop).
        task (str): The initial prompt or question to start the debate.
        agents (list[Agent]): A list containing exactly two Agent instances who will debate.
        img (str, optional): An optional image input to be passed to each agent's run method.
        output_type (str): The format for the output conversation history. Passed to
            `history_output_formatter`. Default is "str-all-except-first".

    Returns:
        list: The formatted conversation history, as produced by `history_output_formatter`.
              The format depends on the `output_type` argument.

    Raises:
        ValueError: If the `agents` list does not contain exactly two Agent instances.
    """
    conversation = Conversation()

    if len(agents) != 2:
        raise ValueError(
            "There must be exactly two agents in the dialogue."
        )

    agent1 = agents[0]
    agent2 = agents[1]

    message = task
    speaker = agent1
    other = agent2

    for i in range(max_loops):
        # Current speaker responds
        response = speaker.run(task=message, img=img)
        conversation.add(speaker.agent_name, response)

        # Swap roles
        message = response
        speaker, other = other, speaker

    return history_output_formatter(
        conversation=conversation, type=output_type
    )
