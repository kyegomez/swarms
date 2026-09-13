from typing import Callable, Union

from swarms.structs.agent import Agent
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
    send_intros: bool = False,
    use_agent_answer: bool = True,
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
        send_intros (bool): If True, tell each agent who they are debating before the
            first turn. Default is False.
        use_agent_answer (bool): If True, extract each agent's final message via
            `agent_answer` before logging and forwarding it; if False, use the raw
            return value of `Agent.run`. Default is True.

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

    agent1, agent2 = agents

    if send_intros:
        agent1_intro = f"You are {agent1.agent_name} debating against {agent2.agent_name}. Your role is to engage in a thoughtful debate."
        agent2_intro = f"You are {agent2.agent_name} debating against {agent1.agent_name}. Your role is to engage in a thoughtful debate."
        agent1.run(task=agent1_intro)
        agent2.run(task=agent2_intro)

    message = task

    for i in range(max_loops):
        speaker = agent1 if i % 2 == 0 else agent2
        response = speaker.run(task=message, img=img)
        answer = (
            agent_answer(speaker, fallback=response)
            if use_agent_answer
            else response
        )
        conversation.add(speaker.agent_name, answer)
        message = answer

    return history_output_formatter(
        conversation=conversation, type=output_type
    )
