"""
InterviewSeries — Conduct a structured interview with follow-up questions.

This structure previously lived in `swarms.structs.multi_agent_debates`. It was
moved here because it is a scripted conversation pattern rather than core
framework machinery: it is built entirely from public `Agent` and `Conversation`
APIs, so it belongs with the examples you copy and adapt.

Copy this file, change the agents and prompts, and run it directly:

    python examples/multi_agent/alternate_debates/interview_series.py
"""

from typing import List

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)


class InterviewSeries:
    """
    Conduct a structured interview with follow-up questions.
    """

    def __init__(
        self,
        questions: List[str] = None,
        interviewer: Agent = None,
        interviewee: Agent = None,
        follow_up_depth: int = 2,
        output_type: str = "str-all-except-first",
    ):
        """
        Initialize the interview series structure.

        Args:
            questions (List[str]): List of prepared interview questions.
            interviewer (Agent): The interviewer agent.
            interviewee (Agent): The interviewee agent.
            follow_up_depth (int): Number of follow-up questions per main question.
            output_type (str): Output format for conversation history.
        """
        self.questions = questions
        self.interviewer = interviewer
        self.interviewee = interviewee
        self.follow_up_depth = follow_up_depth
        self.output_type = output_type

    def run(self, task: str):
        """
        Execute the interview series.

        Args:
            task (str): The main interview topic or context.

        Returns:
            list: Formatted conversation history.
        """
        conversation = Conversation()

        if not self.interviewer or not self.interviewee:
            raise ValueError(
                "Both interviewer and interviewee agents are required."
            )

        if not self.questions:
            self.questions = [
                "Tell me about yourself.",
                "What are your main interests?",
                "What are your goals?",
            ]

        # Inform both agents about their roles
        interviewer_intro = f"You are {self.interviewer.agent_name}, conducting an interview with {self.interviewee.agent_name}. Ask thoughtful questions and follow up appropriately."
        interviewee_intro = f"You are {self.interviewee.agent_name}, being interviewed by {self.interviewer.agent_name}. Provide detailed and honest responses."

        self.interviewer.run(task=interviewer_intro)
        self.interviewee.run(task=interviewee_intro)

        for question in self.questions:
            # Ask main question
            interviewer_response = self.interviewer.run(
                task=f"Ask this question: {question}"
            )
            conversation.add(
                self.interviewer.agent_name, interviewer_response
            )

            # Interviewee responds
            interviewee_response = self.interviewee.run(
                task=interviewer_response
            )
            conversation.add(
                self.interviewee.agent_name, interviewee_response
            )

            # Follow-up questions
            for follow_up in range(self.follow_up_depth):
                follow_up_prompt = f"Based on the response '{interviewee_response}', ask a relevant follow-up question."
                follow_up_question = self.interviewer.run(
                    task=follow_up_prompt
                )
                conversation.add(
                    self.interviewer.agent_name, follow_up_question
                )

                follow_up_response = self.interviewee.run(
                    task=follow_up_question
                )
                conversation.add(
                    self.interviewee.agent_name, follow_up_response
                )

        return history_output_formatter(
            conversation=conversation, type=self.output_type
        )


if __name__ == "__main__":
    interviewer = Agent(
        agent_name="Tech-Journalist",
        agent_description="Technology journalist conducting a founder interview",
        system_prompt="You are a technology journalist. Ask sharp, specific questions and follow up on vague answers.",
        model_name="gpt-5.4",
        max_loops=1,
    )
    interviewee = Agent(
        agent_name="AI-Founder",
        agent_description="Founder of an AI infrastructure startup",
        system_prompt="You are the founder of an AI infrastructure startup. Answer candidly and concretely.",
        model_name="gpt-5.4",
        max_loops=1,
    )

    interview = InterviewSeries(
        questions=[
            "What problem does your company actually solve?",
            "How do you differentiate from the incumbents?",
            "What is the hardest technical problem you have had to solve?",
        ],
        interviewer=interviewer,
        interviewee=interviewee,
        follow_up_depth=2,
    )

    print(
        interview.run(
            "Founder interview on building AI infrastructure"
        )
    )
