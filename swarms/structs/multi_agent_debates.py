from typing import Callable, Union, List

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
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
        agents: list[Union[Agent, Callable]] = None,
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
        conversation = Conversation()

        if len(self.agents) != 2:
            raise ValueError(
                "There must be exactly two agents in the dialogue."
            )

        agent1 = self.agents[0]
        agent2 = self.agents[1]

        # Inform agents about each other
        agent1_intro = f"You are {agent1.agent_name} debating against {agent2.agent_name}. Your role is to engage in a thoughtful debate."
        agent2_intro = f"You are {agent2.agent_name} debating against {agent1.agent_name}. Your role is to engage in a thoughtful debate."

        # Set up initial context for both agents
        agent1.run(task=agent1_intro)
        agent2.run(task=agent2_intro)

        message = task
        speaker = agent1
        other = agent2

        for i in range(self.max_loops):
            # Current speaker responds
            response = speaker.run(task=message, img=self.img)
            conversation.add(speaker.agent_name, response)

            # Swap roles
            message = response
            speaker, other = other, speaker

        return history_output_formatter(
            conversation=conversation, type=self.output_type
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
            synthesis_prompt = f"Synthesize the expert responses and ask a follow-up question: {[msg['content'] for msg in conversation.conversation_history[-len(self.agents):]]}"
            synthesis = self.moderator.run(task=synthesis_prompt)
            conversation.add(self.moderator.agent_name, synthesis)

            current_topic = synthesis

        return history_output_formatter(
            conversation=conversation, type=self.output_type
        )


class RoundTableDiscussion:
    """
    Simulate a round table where each participant speaks in order, then the cycle repeats.
    """

    def __init__(
        self,
        max_cycles: int = 2,
        agents: List[Agent] = None,
        facilitator: Agent = None,
        output_type: str = "str-all-except-first",
    ):
        """
        Initialize the round table discussion structure.

        Args:
            max_cycles (int): Number of complete speaking cycles.
            agents (List[Agent]): List of participants in the round table.
            facilitator (Agent): The facilitator agent who manages the discussion.
            output_type (str): Output format for conversation history.
        """
        self.max_cycles = max_cycles
        self.agents = agents
        self.facilitator = facilitator
        self.output_type = output_type

    def run(self, task: str):
        """
        Execute the round table discussion.

        Args:
            task (str): The main agenda item for discussion.

        Returns:
            list: Formatted conversation history.
        """
        conversation = Conversation()

        if not self.agents or len(self.agents) < 2:
            raise ValueError(
                "At least two participants are required for round table discussion."
            )

        if not self.facilitator:
            raise ValueError(
                "A facilitator agent is required for round table discussion."
            )

        # Create participant list for context
        participant_names = [
            agent.agent_name for agent in self.agents
        ]
        participant_list = f"Round table participants: {', '.join(participant_names)}. Facilitator: {self.facilitator.agent_name}."

        # Inform facilitator about all participants
        facilitator_intro = f"You are {self.facilitator.agent_name}, facilitating a round table discussion. {participant_list} Ensure everyone gets equal speaking time."
        self.facilitator.run(task=facilitator_intro)

        # Inform each participant about the round table setup
        for i, participant in enumerate(self.agents):
            other_participants = [
                name
                for j, name in enumerate(participant_names)
                if j != i
            ]
            participant_intro = f"You are {participant.agent_name}, Participant {i+1} in this round table. Other participants: {', '.join(other_participants)}. Facilitator: {self.facilitator.agent_name}. Share your thoughts when called upon."
            participant.run(task=participant_intro)

        current_agenda = task

        for cycle in range(self.max_cycles):
            # Facilitator introduces the cycle
            cycle_intro = f"Cycle {cycle + 1}: {current_agenda}"
            facilitator_response = self.facilitator.run(
                task=cycle_intro
            )
            conversation.add(
                self.facilitator.agent_name, facilitator_response
            )

            # Each participant speaks in order
            for i, participant in enumerate(self.agents):
                participant_prompt = f"Participant {participant.agent_name}, please share your thoughts on: {facilitator_response}"
                participant_response = participant.run(
                    task=participant_prompt
                )
                conversation.add(
                    participant.agent_name, participant_response
                )

            # Facilitator summarizes and sets next agenda
            summary_prompt = f"Summarize the round and set the next agenda item: {[msg['content'] for msg in conversation.conversation_history[-len(self.agents):]]}"
            summary = self.facilitator.run(task=summary_prompt)
            conversation.add(self.facilitator.agent_name, summary)

            current_agenda = summary

        return history_output_formatter(
            conversation=conversation, type=self.output_type
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


class TrialSimulation:
    """
    Simulate a legal trial with structured phases and roles.
    """

    def __init__(
        self,
        prosecution: Agent = None,
        defense: Agent = None,
        judge: Agent = None,
        witnesses: List[Agent] = None,
        phases: List[str] = None,
        output_type: str = "str-all-except-first",
    ):
        """
        Initialize the trial simulation structure.

        Args:
            prosecution (Agent): The prosecution attorney agent.
            defense (Agent): The defense attorney agent.
            judge (Agent): The judge agent who presides over the trial.
            witnesses (List[Agent]): List of witness agents.
            phases (List[str]): List of trial phases to simulate.
            output_type (str): Output format for conversation history.
        """
        self.prosecution = prosecution
        self.defense = defense
        self.judge = judge
        self.witnesses = witnesses
        self.phases = phases
        self.output_type = output_type

    def run(self, task: str):
        """
        Execute the trial simulation.

        Args:
            task (str): Description of the legal case.

        Returns:
            list: Formatted conversation history.
        """
        conversation = Conversation()

        if not self.prosecution or not self.defense or not self.judge:
            raise ValueError(
                "Prosecution, defense, and judge agents are all required."
            )

        if not self.phases:
            self.phases = ["opening", "testimony", "cross", "closing"]

        # Create trial participant list for context
        witness_names = [
            witness.agent_name for witness in (self.witnesses or [])
        ]
        trial_participants = f"Prosecution: {self.prosecution.agent_name}. Defense: {self.defense.agent_name}. Judge: {self.judge.agent_name}."
        if witness_names:
            trial_participants += (
                f" Witnesses: {', '.join(witness_names)}."
            )

        # Inform judge about all participants
        judge_intro = f"You are {self.judge.agent_name}, presiding over this trial. {trial_participants} Maintain order and ensure proper legal procedure."
        self.judge.run(task=judge_intro)

        # Inform prosecution about trial setup
        prosecution_intro = f"You are {self.prosecution.agent_name}, prosecuting attorney. {trial_participants} Present the case for the prosecution professionally."
        self.prosecution.run(task=prosecution_intro)

        # Inform defense about trial setup
        defense_intro = f"You are {self.defense.agent_name}, defense attorney. {trial_participants} Defend your client professionally."
        self.defense.run(task=defense_intro)

        # Inform witnesses about their role
        for witness in self.witnesses or []:
            witness_intro = f"You are {witness.agent_name}, a witness in this trial. {trial_participants} Provide truthful testimony when called."
            witness.run(task=witness_intro)

        current_case = task

        for phase in self.phases:
            # Judge opens the phase
            phase_opening = (
                f"Phase: {phase.upper()}. Case: {current_case}"
            )
            judge_opening = self.judge.run(task=phase_opening)
            conversation.add(self.judge.agent_name, judge_opening)

            if phase == "opening":
                # Prosecution opening statement
                prosecution_opening = self.prosecution.run(
                    task=f"Give opening statement for: {current_case}"
                )
                conversation.add(
                    self.prosecution.agent_name, prosecution_opening
                )

                # Defense opening statement
                defense_opening = self.defense.run(
                    task=f"Give opening statement responding to: {prosecution_opening}"
                )
                conversation.add(
                    self.defense.agent_name, defense_opening
                )

            elif phase == "testimony" and self.witnesses:
                # Witness testimony
                for i, witness in enumerate(self.witnesses):
                    witness_testimony = witness.run(
                        task=f"Provide testimony for: {current_case}"
                    )
                    conversation.add(
                        witness.agent_name, witness_testimony
                    )

            elif phase == "cross":
                # Cross-examination
                for witness in self.witnesses or []:
                    cross_exam = self.prosecution.run(
                        task=f"Cross-examine this testimony: {witness_testimony}"
                    )
                    conversation.add(
                        self.prosecution.agent_name, cross_exam
                    )

                    redirect = self.defense.run(
                        task=f"Redirect examination: {cross_exam}"
                    )
                    conversation.add(
                        self.defense.agent_name, redirect
                    )

            elif phase == "closing":
                # Closing arguments
                prosecution_closing = self.prosecution.run(
                    task="Give closing argument"
                )
                conversation.add(
                    self.prosecution.agent_name, prosecution_closing
                )

                defense_closing = self.defense.run(
                    task=f"Give closing argument responding to: {prosecution_closing}"
                )
                conversation.add(
                    self.defense.agent_name, defense_closing
                )

                # Judge's verdict
                verdict_prompt = f"Render verdict based on: {[msg['content'] for msg in conversation.conversation_history[-2:]]}"
                verdict = self.judge.run(task=verdict_prompt)
                conversation.add(self.judge.agent_name, verdict)

        return history_output_formatter(
            conversation=conversation, type=self.output_type
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
