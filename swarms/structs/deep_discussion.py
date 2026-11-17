from typing import Callable, Union

from swarms.structs.agent import Agent, reset_agents
from swarms.structs.context_utils import agent_answer
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

    Both agents start from a clean history, and each one's first turn opens
    with who it is debating, so a reused pair does not resend the last debate.

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
    reset_agents(agents)

    message = task
    speaker = agent1
    other = agent2

    for i in range(max_loops):
        prompt = message
        if i < 2:
            prompt = (
                f"You are {speaker.agent_name} debating against "
                f"{other.agent_name}. Your role is to engage in a "
                f"thoughtful debate.\n\n{message}"
            )

        # Current speaker responds
        response = speaker.run(task=prompt, img=img)
        answer = agent_answer(speaker, fallback=response)
        conversation.add(speaker.agent_name, answer)

        # Swap roles
        message = answer
        speaker, other = other, speaker

    return history_output_formatter(
        conversation=conversation, type=output_type
    )
