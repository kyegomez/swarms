"""
MentorshipSession — Simulate a mentorship session with structured learning and feedback.

This structure previously lived in `swarms.structs.multi_agent_debates`. It was
moved here because it is a scripted conversation pattern rather than core
framework machinery: it is built entirely from public `Agent` and `Conversation`
APIs, so it belongs with the examples you copy and adapt.

Copy this file, change the agents and prompts, and run it directly:

    python examples/multi_agent/alternate_debates/mentorship_session.py
"""

from typing import List

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)


class MentorshipSession:
    """
    Simulate a mentorship session with structured learning and feedback.
    """

    def __init__(
        self,
        mentor: Agent = None,
        mentee: Agent = None,
        session_count: int = 3,
        include_feedback: bool = True,
        output_type: str = "str-all-except-first",
    ):
        """
        Initialize the mentorship session structure.

        Args:
            mentor (Agent): The mentor agent who provides guidance.
            mentee (Agent): The mentee agent who is learning.
            session_count (int): Number of mentorship sessions.
            include_feedback (bool): Whether to include feedback in the sessions.
            output_type (str): Output format for conversation history.
        """
        self.mentor = mentor
        self.mentee = mentee
        self.session_count = session_count
        self.include_feedback = include_feedback
        self.output_type = output_type

    def run(self, task: str):
        """
        Execute the mentorship session.

        Args:
            task (str): The learning objective for the mentorship.

        Returns:
            list: Formatted conversation history.
        """
        conversation = Conversation()

        if not self.mentor or not self.mentee:
            raise ValueError(
                "Both mentor and mentee agents are required."
            )

        if not task:
            task = "Professional development and skill improvement"

        # Inform both agents about their roles
        mentor_intro = f"You are {self.mentor.agent_name}, mentoring {self.mentee.agent_name}. Provide guidance, support, and constructive feedback."
        mentee_intro = f"You are {self.mentee.agent_name}, being mentored by {self.mentor.agent_name}. Ask questions, share progress, and be open to feedback."

        self.mentor.run(task=mentor_intro)
        self.mentee.run(task=mentee_intro)

        current_goal = task

        for session in range(self.session_count):
            # Mentor opens the session
            session_opening = (
                f"Session {session + 1}: Let's work on {current_goal}"
            )
            mentor_opening = self.mentor.run(task=session_opening)
            conversation.add(self.mentor.agent_name, mentor_opening)

            # Mentee shares progress and asks questions
            mentee_prompt = f"Mentee {self.mentee.agent_name}, share your progress and ask questions about: {current_goal}"
            mentee_update = self.mentee.run(task=mentee_prompt)
            conversation.add(self.mentee.agent_name, mentee_update)

            # Mentor provides guidance
            guidance_prompt = f"Mentor {self.mentor.agent_name}, provide guidance based on: {mentee_update}"
            mentor_guidance = self.mentor.run(task=guidance_prompt)
            conversation.add(self.mentor.agent_name, mentor_guidance)

            if self.include_feedback:
                # Mentee asks for specific feedback
                feedback_request = self.mentee.run(
                    task="Ask for specific feedback on your progress"
                )
                conversation.add(
                    self.mentee.agent_name, feedback_request
                )

                # Mentor provides detailed feedback
                detailed_feedback = self.mentor.run(
                    task=f"Provide detailed feedback on: {feedback_request}"
                )
                conversation.add(
                    self.mentor.agent_name, detailed_feedback
                )

            # Set next session goal
            next_goal_prompt = "Set the goal for the next session based on this discussion."
            next_goal = self.mentor.run(task=next_goal_prompt)
            conversation.add(self.mentor.agent_name, next_goal)

            current_goal = next_goal

        return history_output_formatter(
            conversation=conversation, type=self.output_type
        )


if __name__ == "__main__":
    mentor = Agent(
        agent_name="Startup-Mentor",
        agent_description="Experienced startup founder and mentor",
        system_prompt="You are a successful startup founder. Give practical, specific guidance and honest feedback.",
        model_name="gpt-5.4",
        max_loops=1,
    )
    mentee = Agent(
        agent_name="Startup-Founder",
        agent_description="Early-stage founder seeking guidance",
        system_prompt="You are an early-stage founder. Share progress candidly and ask focused questions.",
        model_name="gpt-5.4",
        max_loops=1,
    )

    mentorship = MentorshipSession(
        mentor=mentor,
        mentee=mentee,
        session_count=2,
        include_feedback=True,
    )

    print(
        mentorship.run(
            "Finding product-market fit for a B2B healthcare AI platform"
        )
    )
