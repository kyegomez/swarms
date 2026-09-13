from typing import List

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.structs.deep_discussion import one_on_one_debate
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)


class OneOnOneDebate:
    """
    Simulate a turn-based debate between two agents for a specified number of loops.
    """

    def __init__(
        self,
        max_loops: int = 1,
        agents: list[Agent] = None,
        img: str = None,
        output_type: str = "str-all-except-first",
    ):
        """
        Initialize the one-on-one debate structure.

        Args:
            max_loops (int): The number of conversational turns (each agent speaks per loop).
            agents (list[Agent]): A list containing exactly two Agent instances who will debate.
            img (str, optional): An optional image input to be passed to each agent's run method.
            output_type (str): The format for the output conversation history.
        """
        self.max_loops = max_loops
        self.agents = agents
        self.img = img
        self.output_type = output_type

    def run(self, task: str):
        """
        Execute the one-on-one debate.

        Args:
            task (str): The initial prompt or question to start the debate.

        Returns:
            list: The formatted conversation history.

        Raises:
            ValueError: If the `agents` list does not contain exactly two Agent instances.
        """
        return one_on_one_debate(
            max_loops=self.max_loops,
            task=task,
            agents=self.agents,
            img=self.img,
            output_type=self.output_type,
            send_intros=True,
            use_agent_answer=False,
        )


class ExpertPanelDiscussion:
    """
    Simulate an expert panel discussion with a moderator guiding the conversation.
    """

    def __init__(
        self,
        max_rounds: int = 3,
        agents: List[Agent] = None,
        moderator: Agent = None,
        output_type: str = "str-all-except-first",
    ):
        """
        Initialize the expert panel discussion structure.

        Args:
            max_rounds (int): Number of discussion rounds.
            agents (List[Agent]): List of expert agents participating in the panel.
            moderator (Agent): The moderator agent who guides the discussion.
            output_type (str): Output format for conversation history.
        """
        self.max_rounds = max_rounds
        self.agents = agents
        self.moderator = moderator
        self.output_type = output_type

    def run(self, task: str):
        """
        Execute the expert panel discussion.

        Args:
            task (str): The main topic for discussion.

        Returns:
            list: Formatted conversation history.
        """
        conversation = Conversation()

        if not self.agents or len(self.agents) < 2:
            raise ValueError(
                "At least two expert agents are required for a panel discussion."
            )

        if not self.moderator:
            raise ValueError(
                "A moderator agent is required for panel discussion."
            )

        # Create participant list for context
        expert_names = [agent.agent_name for agent in self.agents]
        participant_list = f"Panel participants: {', '.join(expert_names)}. Moderator: {self.moderator.agent_name}."

        # Inform moderator about all participants
        moderator_intro = f"You are {self.moderator.agent_name}, moderating a panel discussion. {participant_list} Guide the discussion professionally."
        self.moderator.run(task=moderator_intro)

        # Inform each expert about the panel setup
        for i, expert in enumerate(self.agents):
            other_experts = [
                name for j, name in enumerate(expert_names) if j != i
            ]
            expert_intro = f"You are {expert.agent_name}, Expert {i+1} on this panel. Other experts: {', '.join(other_experts)}. Moderator: {self.moderator.agent_name}. Provide expert insights."
            expert.run(task=expert_intro)

        current_topic = task

        for round_num in range(self.max_rounds):
            # Moderator introduces the round
            moderator_prompt = (
                f"Round {round_num + 1}: {current_topic}"
            )
            moderator_response = self.moderator.run(
                task=moderator_prompt
            )
            conversation.add(
                self.moderator.agent_name, moderator_response
            )

            # Each expert responds
            for i, expert in enumerate(self.agents):
                expert_prompt = f"Expert {expert.agent_name}, please respond to: {moderator_response}"
                expert_response = expert.run(task=expert_prompt)
                conversation.add(expert.agent_name, expert_response)

            # Moderator synthesizes and asks follow-up
            panel = conversation.conversation_history[
                -len(self.agents) :
            ]
            transcript = "\n\n".join(
                f"{msg['role']}: {msg['content']}" for msg in panel
            )
            synthesis_prompt = (
                "Synthesize the expert responses below and ask a "
                f"follow-up question.\n\n{transcript}"
            )
            synthesis = self.moderator.run(task=synthesis_prompt)
            conversation.add(self.moderator.agent_name, synthesis)

            current_topic = synthesis

        return history_output_formatter(
            conversation=conversation, type=self.output_type
        )
