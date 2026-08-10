"""
NegotiationSession — Simulate a negotiation with multiple parties working toward agreement.

This structure previously lived in `swarms.structs.multi_agent_debates`. It was
moved here because it is a scripted conversation pattern rather than core
framework machinery: it is built entirely from public `Agent` and `Conversation`
APIs, so it belongs with the examples you copy and adapt.

Copy this file, change the agents and prompts, and run it directly:

    python examples/multi_agent/alternate_debates/negotiation_session.py
"""

from typing import List

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)


class NegotiationSession:
    """
    Simulate a negotiation with multiple parties working toward agreement.
    """

    def __init__(
        self,
        parties: List[Agent] = None,
        mediator: Agent = None,
        negotiation_rounds: int = 5,
        include_concessions: bool = True,
        output_type: str = "str-all-except-first",
    ):
        """
        Initialize the negotiation session structure.

        Args:
            parties (List[Agent]): List of negotiating parties.
            mediator (Agent): The mediator who facilitates the negotiation.
            negotiation_rounds (int): Number of negotiation rounds.
            include_concessions (bool): Whether parties can make concessions.
            output_type (str): Output format for conversation history.
        """
        self.parties = parties
        self.mediator = mediator
        self.negotiation_rounds = negotiation_rounds
        self.include_concessions = include_concessions
        self.output_type = output_type

    def run(self, task: str):
        """
        Execute the negotiation session.

        Args:
            task (str): The terms or issues to be negotiated.

        Returns:
            list: Formatted conversation history.
        """
        conversation = Conversation()

        if not self.parties or len(self.parties) < 2:
            raise ValueError(
                "At least two parties are required for negotiation."
            )

        if not self.mediator:
            raise ValueError(
                "A mediator agent is required for negotiation session."
            )

        # Create party list for context
        party_names = [party.agent_name for party in self.parties]
        party_list = f"Negotiating parties: {', '.join(party_names)}. Mediator: {self.mediator.agent_name}."

        # Inform mediator about all parties
        mediator_intro = f"You are {self.mediator.agent_name}, mediating a negotiation. {party_list} Facilitate productive discussion and help reach agreement."
        self.mediator.run(task=mediator_intro)

        # Inform each party about the negotiation setup
        for i, party in enumerate(self.parties):
            other_parties = [
                name for j, name in enumerate(party_names) if j != i
            ]
            party_intro = f"You are {party.agent_name}, Party {i+1} in this negotiation. Other parties: {', '.join(other_parties)}. Mediator: {self.mediator.agent_name}. Present your position clearly and be willing to compromise."
            party.run(task=party_intro)

        current_terms = task

        for round_num in range(self.negotiation_rounds):
            # Mediator opens the round
            round_opening = (
                f"Negotiation Round {round_num + 1}: {current_terms}"
            )
            mediator_opening = self.mediator.run(task=round_opening)
            conversation.add(
                self.mediator.agent_name, mediator_opening
            )

            # Each party presents their position
            for i, party in enumerate(self.parties):
                position_prompt = f"Party {party.agent_name}, present your position on: {current_terms}"
                party_position = party.run(task=position_prompt)
                conversation.add(party.agent_name, party_position)

            # Parties respond to each other's positions
            all_positions = [
                msg["content"]
                for msg in conversation.conversation_history[
                    -len(self.parties) :
                ]
            ]
            for i, party in enumerate(self.parties):
                response_prompt = f"Party {party.agent_name}, respond to the other positions: {all_positions}"
                party_response = party.run(task=response_prompt)
                conversation.add(party.agent_name, party_response)

            if self.include_concessions:
                # Parties make concessions
                for i, party in enumerate(self.parties):
                    concession_prompt = f"Party {party.agent_name}, consider making a concession based on the discussion."
                    party_concession = party.run(
                        task=concession_prompt
                    )
                    conversation.add(
                        party.agent_name, party_concession
                    )

            # Mediator summarizes and proposes next steps
            summary_prompt = "Summarize the round and propose next steps for agreement."
            mediator_summary = self.mediator.run(task=summary_prompt)
            conversation.add(
                self.mediator.agent_name, mediator_summary
            )

            current_terms = mediator_summary

        return history_output_formatter(
            conversation=conversation, type=self.output_type
        )


if __name__ == "__main__":
    parties = [
        Agent(
            agent_name="Buyer",
            agent_description="Acquiring company representative",
            system_prompt="You represent the buyer. Push for favorable terms while keeping the deal alive.",
            model_name="gpt-5.4",
            max_loops=1,
        ),
        Agent(
            agent_name="Seller",
            agent_description="Selling company representative",
            system_prompt="You represent the seller. Maximize valuation and protect your team's interests.",
            model_name="gpt-5.4",
            max_loops=1,
        ),
    ]
    mediator = Agent(
        agent_name="Deal-Mediator",
        agent_description="Neutral deal mediator",
        system_prompt="You mediate the negotiation. Find the zone of possible agreement and drive toward terms.",
        model_name="gpt-5.4",
        max_loops=1,
    )

    negotiation = NegotiationSession(
        parties=parties,
        mediator=mediator,
        negotiation_rounds=2,
        include_concessions=True,
    )

    print(
        negotiation.run(
            "Acquisition terms: purchase price, earnout structure, and retention packages "
            "for the 40-person engineering team."
        )
    )
