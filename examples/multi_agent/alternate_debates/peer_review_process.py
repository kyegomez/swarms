"""
PeerReviewProcess — Simulate academic peer review with multiple reviewers and author responses.

This structure previously lived in `swarms.structs.multi_agent_debates`. It was
moved here because it is a scripted conversation pattern rather than core
framework machinery: it is built entirely from public `Agent` and `Conversation`
APIs, so it belongs with the examples you copy and adapt.

Copy this file, change the agents and prompts, and run it directly:

    python examples/multi_agent/alternate_debates/peer_review_process.py
"""

from typing import List

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)


class PeerReviewProcess:
    """
    Simulate academic peer review with multiple reviewers and author responses.
    """

    def __init__(
        self,
        reviewers: List[Agent] = None,
        author: Agent = None,
        review_rounds: int = 2,
        output_type: str = "str-all-except-first",
    ):
        """
        Initialize the peer review process structure.

        Args:
            reviewers (List[Agent]): List of reviewer agents.
            author (Agent): The author agent who responds to reviews.
            review_rounds (int): Number of review rounds.
            output_type (str): Output format for conversation history.
        """
        self.reviewers = reviewers
        self.author = author
        self.review_rounds = review_rounds
        self.output_type = output_type

    def run(self, task: str):
        """
        Execute the peer review process.

        Args:
            task (str): The work being reviewed.

        Returns:
            list: Formatted conversation history.
        """
        conversation = Conversation()

        if not self.reviewers or len(self.reviewers) < 1:
            raise ValueError("At least one reviewer is required.")

        if not self.author:
            raise ValueError(
                "An author agent is required for peer review."
            )

        # Create reviewer list for context
        reviewer_names = [
            reviewer.agent_name for reviewer in self.reviewers
        ]
        reviewer_list = f"Reviewers: {', '.join(reviewer_names)}. Author: {self.author.agent_name}."

        # Inform author about all reviewers
        author_intro = f"You are {self.author.agent_name}, the author of the work being reviewed. {reviewer_list} Respond professionally to feedback."
        self.author.run(task=author_intro)

        # Inform each reviewer about the review process
        for i, reviewer in enumerate(self.reviewers):
            other_reviewers = [
                name
                for j, name in enumerate(reviewer_names)
                if j != i
            ]
            reviewer_intro = f"You are {reviewer.agent_name}, Reviewer {i+1}. Other reviewers: {', '.join(other_reviewers)}. Author: {self.author.agent_name}. Provide constructive feedback."
            reviewer.run(task=reviewer_intro)

        current_submission = task

        for round_num in range(self.review_rounds):
            # Each reviewer provides feedback
            for i, reviewer in enumerate(self.reviewers):
                review_prompt = f"Reviewer {reviewer.agent_name}, please review this work: {current_submission}"
                review_feedback = reviewer.run(task=review_prompt)
                conversation.add(reviewer.agent_name, review_feedback)

            # Author responds to all reviews
            all_reviews = [
                msg["content"]
                for msg in conversation.conversation_history[
                    -len(self.reviewers) :
                ]
            ]
            author_response_prompt = f"Author {self.author.agent_name}, please respond to these reviews: {all_reviews}"
            author_response = self.author.run(
                task=author_response_prompt
            )
            conversation.add(self.author.agent_name, author_response)

            current_submission = author_response

        return history_output_formatter(
            conversation=conversation, type=self.output_type
        )


if __name__ == "__main__":
    reviewers = [
        Agent(
            agent_name=f"Reviewer-{i}",
            agent_description="Academic peer reviewer",
            system_prompt=f"You are Reviewer {i} for a machine learning venue. Give specific, constructive, critical feedback.",
            model_name="gpt-5.4",
            max_loops=1,
        )
        for i in (1, 2)
    ]
    author = Agent(
        agent_name="Author",
        agent_description="Paper author responding to reviews",
        system_prompt="You are the paper's author. Respond to reviewer concerns precisely and professionally.",
        model_name="gpt-5.4",
        max_loops=1,
    )

    review = PeerReviewProcess(
        reviewers=reviewers, author=author, review_rounds=2
    )

    print(
        review.run(
            "Paper: 'Sparse Mixture-of-Experts Routing for Long-Context Retrieval'. "
            "Claims a 30% latency reduction with no quality loss on three benchmarks."
        )
    )
