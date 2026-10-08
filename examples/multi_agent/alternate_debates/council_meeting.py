"""
CouncilMeeting — Simulate a council meeting with structured discussion and decision-making.

This structure previously lived in `swarms.structs.multi_agent_debates`. It was
moved here because it is a scripted conversation pattern rather than core
framework machinery: it is built entirely from public `Agent` and `Conversation`
APIs, so it belongs with the examples you copy and adapt.

Copy this file, change the agents and prompts, and run it directly:

    python examples/multi_agent/alternate_debates/council_meeting.py
"""

from typing import List

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)


class CouncilMeeting:
    """
    Simulate a council meeting with structured discussion and decision-making.
    """

    def __init__(
        self,
        council_members: List[Agent] = None,
        chairperson: Agent = None,
        voting_rounds: int = 1,
        require_consensus: bool = False,
        output_type: str = "str-all-except-first",
    ):
        """
        Initialize the council meeting structure.

        Args:
            council_members (List[Agent]): List of council member agents.
            chairperson (Agent): The chairperson who manages the meeting.
            voting_rounds (int): Number of voting rounds.
            require_consensus (bool): Whether consensus is required for approval.
            output_type (str): Output format for conversation history.
        """
        self.council_members = council_members
        self.chairperson = chairperson
        self.voting_rounds = voting_rounds
        self.require_consensus = require_consensus
        self.output_type = output_type

    def run(self, task: str):
        """
        Execute the council meeting.

        Args:
            task (str): The proposal to be discussed and voted on.

        Returns:
            list: Formatted conversation history.
        """
        conversation = Conversation()

        if not self.council_members or len(self.council_members) < 2:
            raise ValueError(
                "At least two council members are required."
            )

        if not self.chairperson:
            raise ValueError(
                "A chairperson agent is required for council meeting."
            )

        # Create council member list for context
        member_names = [
            member.agent_name for member in self.council_members
        ]
        council_list = f"Council members: {', '.join(member_names)}. Chairperson: {self.chairperson.agent_name}."

        # Inform chairperson about all members
        chairperson_intro = f"You are {self.chairperson.agent_name}, chairing this council meeting. {council_list} Manage the discussion and voting process professionally."
        self.chairperson.run(task=chairperson_intro)

        # Inform each council member about the meeting setup
        for i, member in enumerate(self.council_members):
            other_members = [
                name for j, name in enumerate(member_names) if j != i
            ]
            member_intro = f"You are {member.agent_name}, Council Member {i+1}. Other members: {', '.join(other_members)}. Chairperson: {self.chairperson.agent_name}. Participate in discussion and vote on proposals."
            member.run(task=member_intro)

        current_proposal = task

        for round_num in range(self.voting_rounds):
            # Chairperson opens the meeting
            meeting_opening = f"Council Meeting Round {round_num + 1}: {current_proposal}"
            chair_opening = self.chairperson.run(task=meeting_opening)
            conversation.add(
                self.chairperson.agent_name, chair_opening
            )

            # Each member discusses the proposal
            for i, member in enumerate(self.council_members):
                member_prompt = f"Council Member {member.agent_name}, discuss this proposal: {current_proposal}"
                member_discussion = member.run(task=member_prompt)
                conversation.add(member.agent_name, member_discussion)

            # Chairperson facilitates discussion and calls for vote
            all_discussions = [
                msg["content"]
                for msg in conversation.conversation_history[
                    -len(self.council_members) :
                ]
            ]
            vote_prompt = f"Based on these discussions {all_discussions}, call for a vote on the proposal."
            vote_call = self.chairperson.run(task=vote_prompt)
            conversation.add(self.chairperson.agent_name, vote_call)

            # Members vote
            for i, member in enumerate(self.council_members):
                vote_prompt = f"Council Member {member.agent_name}, vote on the proposal (approve/reject/abstain)."
                member_vote = member.run(task=vote_prompt)
                conversation.add(member.agent_name, member_vote)

            # Chairperson announces result
            all_votes = [
                msg["content"]
                for msg in conversation.conversation_history[
                    -len(self.council_members) :
                ]
            ]
            result_prompt = (
                f"Announce the voting result based on: {all_votes}"
            )
            result = self.chairperson.run(task=result_prompt)
            conversation.add(self.chairperson.agent_name, result)

            current_proposal = result

        return history_output_formatter(
            conversation=conversation, type=self.output_type
        )


if __name__ == "__main__":
    council_members = [
        Agent(
            agent_name=f"Council-Member-{i}",
            agent_description="Investment council member",
            system_prompt=f"You are council member {i}. Evaluate proposals on risk, return, and strategic fit, then vote.",
            model_name="gpt-5.4",
            max_loops=1,
        )
        for i in (1, 2, 3)
    ]
    chairperson = Agent(
        agent_name="Chairperson",
        agent_description="Council chairperson",
        system_prompt="You chair the council. Run the discussion, call the vote, and announce the result.",
        model_name="gpt-5.4",
        max_loops=1,
    )

    council = CouncilMeeting(
        council_members=council_members,
        chairperson=chairperson,
        voting_rounds=1,
        require_consensus=False,
    )

    print(
        council.run(
            "Proposal: allocate $5M to a Series A round in an autonomous-logistics startup "
            "at a $40M pre-money valuation."
        )
    )
