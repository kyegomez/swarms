"""
BrainstormingSession — Simulate a brainstorming session where participants build on each other's ideas.

This structure previously lived in `swarms.structs.multi_agent_debates`. It was
moved here because it is a scripted conversation pattern rather than core
framework machinery: it is built entirely from public `Agent` and `Conversation`
APIs, so it belongs with the examples you copy and adapt.

Copy this file, change the agents and prompts, and run it directly:

    python examples/multi_agent/alternate_debates/brainstorming_session.py
"""

from typing import List

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)


class BrainstormingSession:
    """
    Simulate a brainstorming session where participants build on each other's ideas.
    """

    def __init__(
        self,
        participants: List[Agent] = None,
        facilitator: Agent = None,
        idea_rounds: int = 3,
        build_on_ideas: bool = True,
        output_type: str = "str-all-except-first",
    ):
        """
        Initialize the brainstorming session structure.

        Args:
            participants (List[Agent]): List of brainstorming participants.
            facilitator (Agent): The facilitator who guides the session.
            idea_rounds (int): Number of idea generation rounds.
            build_on_ideas (bool): Whether participants should build on previous ideas.
            output_type (str): Output format for conversation history.
        """
        self.participants = participants
        self.facilitator = facilitator
        self.idea_rounds = idea_rounds
        self.build_on_ideas = build_on_ideas
        self.output_type = output_type

    def run(self, task: str):
        """
        Execute the brainstorming session.

        Args:
            task (str): The problem or challenge to brainstorm about.

        Returns:
            list: Formatted conversation history.
        """
        conversation = Conversation()

        if not self.participants or len(self.participants) < 2:
            raise ValueError(
                "At least two participants are required for brainstorming."
            )

        if not self.facilitator:
            raise ValueError(
                "A facilitator agent is required for brainstorming session."
            )

        # Create participant list for context
        participant_names = [
            participant.agent_name
            for participant in self.participants
        ]
        participant_list = f"Brainstorming participants: {', '.join(participant_names)}. Facilitator: {self.facilitator.agent_name}."

        # Inform facilitator about all participants
        facilitator_intro = f"You are {self.facilitator.agent_name}, facilitating a brainstorming session. {participant_list} Encourage creative thinking and idea building."
        self.facilitator.run(task=facilitator_intro)

        # Inform each participant about the brainstorming setup
        for i, participant in enumerate(self.participants):
            other_participants = [
                name
                for j, name in enumerate(participant_names)
                if j != i
            ]
            participant_intro = f"You are {participant.agent_name}, Participant {i+1} in this brainstorming session. Other participants: {', '.join(other_participants)}. Facilitator: {self.facilitator.agent_name}. Contribute creative ideas and build on others' suggestions."
            participant.run(task=participant_intro)

        current_problem = task
        all_ideas = []

        for round_num in range(self.idea_rounds):
            # Facilitator introduces the round
            round_intro = f"Round {round_num + 1}: Let's brainstorm about {current_problem}"
            facilitator_intro = self.facilitator.run(task=round_intro)
            conversation.add(
                self.facilitator.agent_name, facilitator_intro
            )

            # Each participant contributes ideas
            for i, participant in enumerate(self.participants):
                if self.build_on_ideas and all_ideas:
                    idea_prompt = f"Participant {participant.agent_name}, build on these previous ideas: {all_ideas[-3:]}"
                else:
                    idea_prompt = f"Participant {participant.agent_name}, suggest ideas for: {current_problem}"

                participant_idea = participant.run(task=idea_prompt)
                conversation.add(
                    participant.agent_name, participant_idea
                )
                all_ideas.append(participant_idea)

            # Facilitator synthesizes and reframes
            synthesis_prompt = f"Synthesize the ideas from this round and reframe the problem: {[msg['content'] for msg in conversation.conversation_history[-len(self.participants):]]}"
            synthesis = self.facilitator.run(task=synthesis_prompt)
            conversation.add(self.facilitator.agent_name, synthesis)

            current_problem = synthesis

        return history_output_formatter(
            conversation=conversation, type=self.output_type
        )


if __name__ == "__main__":
    participants = [
        Agent(
            agent_name="Designer",
            agent_description="Product designer",
            system_prompt="You are a product designer. Contribute user-centered, concrete ideas.",
            model_name="gpt-5.4",
            max_loops=1,
        ),
        Agent(
            agent_name="Engineer",
            agent_description="Systems engineer",
            system_prompt="You are a systems engineer. Contribute technically feasible ideas and flag constraints.",
            model_name="gpt-5.4",
            max_loops=1,
        ),
        Agent(
            agent_name="Growth-Lead",
            agent_description="Growth strategist",
            system_prompt="You are a growth strategist. Contribute ideas that drive adoption and retention.",
            model_name="gpt-5.4",
            max_loops=1,
        ),
    ]
    facilitator = Agent(
        agent_name="Facilitator",
        agent_description="Brainstorm facilitator",
        system_prompt="You facilitate brainstorming. Encourage divergent thinking, then synthesize and reframe.",
        model_name="gpt-5.4",
        max_loops=1,
    )

    brainstorm = BrainstormingSession(
        participants=participants,
        facilitator=facilitator,
        idea_rounds=2,
        build_on_ideas=True,
    )

    print(
        brainstorm.run(
            "How can we cut new-user time-to-first-value from 30 minutes to under 5?"
        )
    )
