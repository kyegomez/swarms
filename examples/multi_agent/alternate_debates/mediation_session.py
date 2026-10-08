"""
MediationSession — Simulate a mediation session to resolve conflicts between parties.

This structure previously lived in `swarms.structs.multi_agent_debates`. It was
moved here because it is a scripted conversation pattern rather than core
framework machinery: it is built entirely from public `Agent` and `Conversation`
APIs, so it belongs with the examples you copy and adapt.

Copy this file, change the agents and prompts, and run it directly:

    python examples/multi_agent/alternate_debates/mediation_session.py
"""

from typing import List

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)


class MediationSession:
    """
    Simulate a mediation session to resolve conflicts between parties.
    """

    def __init__(
        self,
        parties: List[Agent] = None,
        mediator: Agent = None,
        max_sessions: int = 3,
        output_type: str = "str-all-except-first",
    ):
        """
        Initialize the mediation session structure.

        Args:
            parties (List[Agent]): List of parties involved in the dispute.
            mediator (Agent): The mediator agent who facilitates resolution.
            max_sessions (int): Number of mediation sessions.
            output_type (str): Output format for conversation history.
        """
        self.parties = parties
        self.mediator = mediator
        self.max_sessions = max_sessions
        self.output_type = output_type

    def run(self, task: str):
        """
        Execute the mediation session.

        Args:
            task (str): Description of the dispute to be mediated.

        Returns:
            list: Formatted conversation history.
        """
        conversation = Conversation()

        if not self.parties or len(self.parties) < 2:
            raise ValueError(
                "At least two parties are required for mediation."
            )

        if not self.mediator:
            raise ValueError(
                "A mediator agent is required for mediation session."
            )

        # Create party list for context
        party_names = [party.agent_name for party in self.parties]
        party_list = f"Disputing parties: {', '.join(party_names)}. Mediator: {self.mediator.agent_name}."

        # Inform mediator about all parties
        mediator_intro = f"You are {self.mediator.agent_name}, mediating a dispute. {party_list} Facilitate resolution fairly and professionally."
        self.mediator.run(task=mediator_intro)

        # Inform each party about the mediation process
        for i, party in enumerate(self.parties):
            other_parties = [
                name for j, name in enumerate(party_names) if j != i
            ]
            party_intro = f"You are {party.agent_name}, Party {i+1} in this mediation. Other parties: {', '.join(other_parties)}. Mediator: {self.mediator.agent_name}. Present your perspective respectfully."
            party.run(task=party_intro)

        current_dispute = task

        for session in range(self.max_sessions):
            # Mediator opens the session
            session_opening = f"Session {session + 1}: Let's address {current_dispute}"
            mediator_opening = self.mediator.run(task=session_opening)
            conversation.add(
                self.mediator.agent_name, mediator_opening
            )

            # Each party presents their perspective
            for i, party in enumerate(self.parties):
                party_prompt = f"Party {party.agent_name}, please share your perspective on: {mediator_opening}"
                party_response = party.run(task=party_prompt)
                conversation.add(party.agent_name, party_response)

            # Mediator facilitates discussion and proposes solutions
            all_perspectives = [
                msg["content"]
                for msg in conversation.conversation_history[
                    -len(self.parties) :
                ]
            ]
            mediation_prompt = f"Based on these perspectives {all_perspectives}, propose a resolution approach."
            mediation_proposal = self.mediator.run(
                task=mediation_prompt
            )
            conversation.add(
                self.mediator.agent_name, mediation_proposal
            )

            current_dispute = mediation_proposal

        return history_output_formatter(
            conversation=conversation, type=self.output_type
        )


if __name__ == "__main__":
    parties = [
        Agent(
            agent_name="Engineering-Lead",
            agent_description="Engineering lead in a resourcing dispute",
            system_prompt="You are the engineering lead. You believe the deadline is unrealistic without more headcount.",
            model_name="gpt-5.4",
            max_loops=1,
        ),
        Agent(
            agent_name="Product-Lead",
            agent_description="Product lead in a resourcing dispute",
            system_prompt="You are the product lead. You believe the launch date is contractually fixed and cannot move.",
            model_name="gpt-5.4",
            max_loops=1,
        ),
    ]
    mediator = Agent(
        agent_name="Mediator",
        agent_description="Neutral mediator",
        system_prompt="You are a neutral mediator. Surface shared interests and propose concrete, workable resolutions.",
        model_name="gpt-5.4",
        max_loops=1,
    )

    mediation = MediationSession(
        parties=parties, mediator=mediator, max_sessions=2
    )

    print(
        mediation.run(
            "Engineering says the Q3 launch needs two more engineers or a four-week slip. "
            "Product says the date is committed to a customer contract."
        )
    )
