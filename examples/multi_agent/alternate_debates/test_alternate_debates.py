"""
Tests for the alternate debate structures in this folder.

These structures moved out of `swarms.structs.multi_agent_debates` into
`examples/`, so their tests moved with them rather than being deleted.

Run from the repo root:

    pytest examples/multi_agent/alternate_debates/test_alternate_debates.py
"""

import sys
from pathlib import Path

import pytest
from loguru import logger

from swarms.structs.agent import Agent

sys.path.insert(0, str(Path(__file__).resolve().parent))

from brainstorming_session import BrainstormingSession  # noqa: E402
from council_meeting import CouncilMeeting  # noqa: E402
from interview_series import InterviewSeries  # noqa: E402
from mediation_session import MediationSession  # noqa: E402
from mentorship_session import MentorshipSession  # noqa: E402
from negotiation_session import NegotiationSession  # noqa: E402
from peer_review_process import PeerReviewProcess  # noqa: E402
from trial_simulation import TrialSimulation  # noqa: E402


def create_function_agent(name: str, system_prompt: str = None):
    if system_prompt is None:
        system_prompt = (
            f"You are {name}. Provide concise and direct responses."
        )

    return Agent(
        agent_name=name,
        agent_description=f"Test agent {name}",
        system_prompt=system_prompt,
        model_name="gpt-5.4",
        max_loops=1,
        verbose=False,
    )


@pytest.fixture
def sample_two_agents():
    return [
        create_function_agent(
            "Agent1", "You are Agent1. Provide concise responses."
        ),
        create_function_agent(
            "Agent2", "You are Agent2. Provide concise responses."
        ),
    ]


@pytest.fixture
def sample_three_agents():
    return [create_function_agent(f"Agent{i}") for i in (1, 2, 3)]


@pytest.fixture
def sample_task():
    return "What is 2+2?"


def test_interview_series_initialization():
    try:
        interviewer = create_function_agent("Interviewer")
        assert interviewer is not None
        interviewee = create_function_agent("Interviewee")
        assert interviewee is not None
        questions = ["Question 1", "Question 2"]
        assert questions is not None
        interview = InterviewSeries(
            questions=questions,
            interviewer=interviewer,
            interviewee=interviewee,
            follow_up_depth=1,
            output_type="str-all-except-first",
        )
        assert interview is not None
        assert interview.questions == questions
        assert interview.interviewer is not None
        assert interview.interviewee is not None
        assert interview.follow_up_depth == 1
        logger.info("InterviewSeries initialization test passed")
    except Exception as e:
        logger.error(
            f"Failed to test InterviewSeries initialization: {e}"
        )
        raise


def test_interview_series_run(sample_task):
    try:
        interviewer = create_function_agent("Interviewer")
        assert interviewer is not None
        interviewee = create_function_agent("Interviewee")
        assert interviewee is not None
        questions = ["Question 1", "Question 2"]
        assert questions is not None
        interview = InterviewSeries(
            questions=questions,
            interviewer=interviewer,
            interviewee=interviewee,
            follow_up_depth=1,
            output_type="str-all-except-first",
        )
        assert interview is not None
        result = interview.run(sample_task)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) >= 0
        logger.info("InterviewSeries run test passed")
    except Exception as e:
        logger.error(f"Failed to test InterviewSeries run: {e}")
        raise


def test_interview_series_no_interviewer(sample_task):
    try:
        interviewee = create_function_agent("Interviewee")
        interview = InterviewSeries(
            questions=["Question 1"],
            interviewer=None,
            interviewee=interviewee,
            follow_up_depth=1,
            output_type="str-all-except-first",
        )
        with pytest.raises(
            ValueError, match="Both interviewer and interviewee"
        ):
            interview.run(sample_task)
        logger.info("InterviewSeries no interviewer test passed")
    except Exception as e:
        logger.error(
            f"Failed to test InterviewSeries no interviewer: {e}"
        )
        raise


def test_interview_series_no_interviewee(sample_task):
    try:
        interviewer = create_function_agent("Interviewer")
        interview = InterviewSeries(
            questions=["Question 1"],
            interviewer=interviewer,
            interviewee=None,
            follow_up_depth=1,
            output_type="str-all-except-first",
        )
        with pytest.raises(
            ValueError, match="Both interviewer and interviewee"
        ):
            interview.run(sample_task)
        logger.info("InterviewSeries no interviewee test passed")
    except Exception as e:
        logger.error(
            f"Failed to test InterviewSeries no interviewee: {e}"
        )
        raise


def test_interview_series_default_questions(sample_task):
    try:
        interviewer = create_function_agent("Interviewer")
        assert interviewer is not None
        interviewee = create_function_agent("Interviewee")
        assert interviewee is not None
        assert sample_task is not None
        interview = InterviewSeries(
            questions=None,
            interviewer=interviewer,
            interviewee=interviewee,
            follow_up_depth=1,
            output_type="str-all-except-first",
        )
        assert interview is not None
        result = interview.run(sample_task)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) >= 0
        logger.info("InterviewSeries default questions test passed")
    except Exception as e:
        logger.error(
            f"Failed to test InterviewSeries default questions: {e}"
        )
        raise


def test_peer_review_process_initialization():
    try:
        reviewers = [
            create_function_agent("Reviewer1"),
            create_function_agent("Reviewer2"),
        ]
        assert reviewers is not None
        assert len(reviewers) == 2
        assert reviewers[0] is not None
        assert reviewers[1] is not None
        author = create_function_agent("Author")
        assert author is not None
        peer_review = PeerReviewProcess(
            reviewers=reviewers,
            author=author,
            review_rounds=2,
            output_type="str-all-except-first",
        )
        assert peer_review is not None
        assert len(peer_review.reviewers) == 2
        assert peer_review.author is not None
        assert peer_review.review_rounds == 2
        logger.info("PeerReviewProcess initialization test passed")
    except Exception as e:
        logger.error(
            f"Failed to test PeerReviewProcess initialization: {e}"
        )
        raise


def test_peer_review_process_run(sample_task):
    try:
        reviewers = [
            create_function_agent("Reviewer1"),
            create_function_agent("Reviewer2"),
        ]
        assert reviewers is not None
        assert len(reviewers) == 2
        author = create_function_agent("Author")
        assert author is not None
        peer_review = PeerReviewProcess(
            reviewers=reviewers,
            author=author,
            review_rounds=2,
            output_type="str-all-except-first",
        )
        assert peer_review is not None
        result = peer_review.run(sample_task)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) >= 0
        logger.info("PeerReviewProcess run test passed")
    except Exception as e:
        logger.error(f"Failed to test PeerReviewProcess run: {e}")
        raise


def test_peer_review_process_no_reviewers(sample_task):
    try:
        author = create_function_agent("Author")
        peer_review = PeerReviewProcess(
            reviewers=[],
            author=author,
            review_rounds=2,
            output_type="str-all-except-first",
        )
        with pytest.raises(ValueError, match="At least one reviewer"):
            peer_review.run(sample_task)
        logger.info("PeerReviewProcess no reviewers test passed")
    except Exception as e:
        logger.error(
            f"Failed to test PeerReviewProcess no reviewers: {e}"
        )
        raise


def test_peer_review_process_no_author(sample_task):
    try:
        reviewers = [create_function_agent("Reviewer1")]
        peer_review = PeerReviewProcess(
            reviewers=reviewers,
            author=None,
            review_rounds=2,
            output_type="str-all-except-first",
        )
        with pytest.raises(
            ValueError, match="author agent is required"
        ):
            peer_review.run(sample_task)
        logger.info("PeerReviewProcess no author test passed")
    except Exception as e:
        logger.error(
            f"Failed to test PeerReviewProcess no author: {e}"
        )
        raise


def test_mediation_session_initialization(sample_two_agents):
    try:
        mediator = create_function_agent("Mediator")
        assert mediator is not None
        assert sample_two_agents is not None
        mediation = MediationSession(
            parties=sample_two_agents,
            mediator=mediator,
            max_sessions=2,
            output_type="str-all-except-first",
        )
        assert mediation is not None
        assert len(mediation.parties) == 2
        assert mediation.mediator is not None
        assert mediation.max_sessions == 2
        logger.info("MediationSession initialization test passed")
    except Exception as e:
        logger.error(
            f"Failed to test MediationSession initialization: {e}"
        )
        raise


def test_mediation_session_run(sample_two_agents, sample_task):
    try:
        mediator = create_function_agent("Mediator")
        assert mediator is not None
        assert sample_two_agents is not None
        mediation = MediationSession(
            parties=sample_two_agents,
            mediator=mediator,
            max_sessions=2,
            output_type="str-all-except-first",
        )
        assert mediation is not None
        result = mediation.run(sample_task)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) >= 0
        logger.info("MediationSession run test passed")
    except Exception as e:
        logger.error(f"Failed to test MediationSession run: {e}")
        raise


def test_mediation_session_insufficient_parties(sample_task):
    try:
        mediator = create_function_agent("Mediator")
        single_party = [create_function_agent("Party1")]
        mediation = MediationSession(
            parties=single_party,
            mediator=mediator,
            max_sessions=2,
            output_type="str-all-except-first",
        )
        with pytest.raises(ValueError, match="At least two parties"):
            mediation.run(sample_task)
        logger.info(
            "MediationSession insufficient parties test passed"
        )
    except Exception as e:
        logger.error(
            f"Failed to test MediationSession insufficient parties: {e}"
        )
        raise


def test_mediation_session_no_mediator(
    sample_two_agents, sample_task
):
    try:
        mediation = MediationSession(
            parties=sample_two_agents,
            mediator=None,
            max_sessions=2,
            output_type="str-all-except-first",
        )
        with pytest.raises(
            ValueError, match="mediator agent is required"
        ):
            mediation.run(sample_task)
        logger.info("MediationSession no mediator test passed")
    except Exception as e:
        logger.error(
            f"Failed to test MediationSession no mediator: {e}"
        )
        raise


def test_brainstorming_session_initialization(sample_three_agents):
    try:
        facilitator = create_function_agent("Facilitator")
        assert facilitator is not None
        assert sample_three_agents is not None
        brainstorming = BrainstormingSession(
            participants=sample_three_agents,
            facilitator=facilitator,
            idea_rounds=2,
            build_on_ideas=True,
            output_type="str-all-except-first",
        )
        assert brainstorming is not None
        assert len(brainstorming.participants) == 3
        assert brainstorming.facilitator is not None
        assert brainstorming.idea_rounds == 2
        assert brainstorming.build_on_ideas is True
        logger.info("BrainstormingSession initialization test passed")
    except Exception as e:
        logger.error(
            f"Failed to test BrainstormingSession initialization: {e}"
        )
        raise


def test_brainstorming_session_run(sample_three_agents, sample_task):
    try:
        facilitator = create_function_agent("Facilitator")
        assert facilitator is not None
        assert sample_three_agents is not None
        brainstorming = BrainstormingSession(
            participants=sample_three_agents,
            facilitator=facilitator,
            idea_rounds=2,
            build_on_ideas=True,
            output_type="str-all-except-first",
        )
        assert brainstorming is not None
        result = brainstorming.run(sample_task)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) >= 0
        logger.info("BrainstormingSession run test passed")
    except Exception as e:
        logger.error(f"Failed to test BrainstormingSession run: {e}")
        raise


def test_brainstorming_session_insufficient_participants(sample_task):
    try:
        facilitator = create_function_agent("Facilitator")
        single_participant = [create_function_agent("Participant1")]
        brainstorming = BrainstormingSession(
            participants=single_participant,
            facilitator=facilitator,
            idea_rounds=2,
            build_on_ideas=True,
            output_type="str-all-except-first",
        )
        with pytest.raises(
            ValueError, match="At least two participants"
        ):
            brainstorming.run(sample_task)
        logger.info(
            "BrainstormingSession insufficient participants test passed"
        )
    except Exception as e:
        logger.error(
            f"Failed to test BrainstormingSession insufficient participants: {e}"
        )
        raise


def test_brainstorming_session_no_facilitator(
    sample_three_agents, sample_task
):
    try:
        brainstorming = BrainstormingSession(
            participants=sample_three_agents,
            facilitator=None,
            idea_rounds=2,
            build_on_ideas=True,
            output_type="str-all-except-first",
        )
        with pytest.raises(
            ValueError, match="facilitator agent is required"
        ):
            brainstorming.run(sample_task)
        logger.info("BrainstormingSession no facilitator test passed")
    except Exception as e:
        logger.error(
            f"Failed to test BrainstormingSession no facilitator: {e}"
        )
        raise


def test_trial_simulation_initialization():
    try:
        prosecution = create_function_agent("Prosecution")
        assert prosecution is not None
        defense = create_function_agent("Defense")
        assert defense is not None
        judge = create_function_agent("Judge")
        assert judge is not None
        witnesses = [create_function_agent("Witness1")]
        assert witnesses is not None
        assert len(witnesses) == 1
        assert witnesses[0] is not None
        trial = TrialSimulation(
            prosecution=prosecution,
            defense=defense,
            judge=judge,
            witnesses=witnesses,
            phases=["opening", "closing"],
            output_type="str-all-except-first",
        )
        assert trial is not None
        assert trial.prosecution is not None
        assert trial.defense is not None
        assert trial.judge is not None
        assert len(trial.witnesses) == 1
        assert trial.phases == ["opening", "closing"]
        logger.info("TrialSimulation initialization test passed")
    except Exception as e:
        logger.error(
            f"Failed to test TrialSimulation initialization: {e}"
        )
        raise


def test_trial_simulation_run(sample_task):
    try:
        prosecution = create_function_agent("Prosecution")
        assert prosecution is not None
        defense = create_function_agent("Defense")
        assert defense is not None
        judge = create_function_agent("Judge")
        assert judge is not None
        trial = TrialSimulation(
            prosecution=prosecution,
            defense=defense,
            judge=judge,
            witnesses=None,
            phases=["opening", "closing"],
            output_type="str-all-except-first",
        )
        assert trial is not None
        result = trial.run(sample_task)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) >= 0
        logger.info("TrialSimulation run test passed")
    except Exception as e:
        logger.error(f"Failed to test TrialSimulation run: {e}")
        raise


def test_trial_simulation_no_prosecution(sample_task):
    try:
        defense = create_function_agent("Defense")
        judge = create_function_agent("Judge")
        trial = TrialSimulation(
            prosecution=None,
            defense=defense,
            judge=judge,
            witnesses=None,
            phases=["opening"],
            output_type="str-all-except-first",
        )
        with pytest.raises(ValueError, match="all required"):
            trial.run(sample_task)
        logger.info("TrialSimulation no prosecution test passed")
    except Exception as e:
        logger.error(
            f"Failed to test TrialSimulation no prosecution: {e}"
        )
        raise


def test_trial_simulation_default_phases(sample_task):
    try:
        prosecution = create_function_agent("Prosecution")
        assert prosecution is not None
        defense = create_function_agent("Defense")
        assert defense is not None
        judge = create_function_agent("Judge")
        assert judge is not None
        assert sample_task is not None
        trial = TrialSimulation(
            prosecution=prosecution,
            defense=defense,
            judge=judge,
            witnesses=None,
            phases=None,
            output_type="str-all-except-first",
        )
        assert trial is not None
        result = trial.run(sample_task)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) >= 0
        logger.info("TrialSimulation default phases test passed")
    except Exception as e:
        logger.error(
            f"Failed to test TrialSimulation default phases: {e}"
        )
        raise


def test_council_meeting_initialization(sample_three_agents):
    try:
        chairperson = create_function_agent("Chairperson")
        assert chairperson is not None
        assert sample_three_agents is not None
        council = CouncilMeeting(
            council_members=sample_three_agents,
            chairperson=chairperson,
            voting_rounds=2,
            require_consensus=False,
            output_type="str-all-except-first",
        )
        assert council is not None
        assert len(council.council_members) == 3
        assert council.chairperson is not None
        assert council.voting_rounds == 2
        assert council.require_consensus is False
        logger.info("CouncilMeeting initialization test passed")
    except Exception as e:
        logger.error(
            f"Failed to test CouncilMeeting initialization: {e}"
        )
        raise


def test_council_meeting_run(sample_three_agents, sample_task):
    try:
        chairperson = create_function_agent("Chairperson")
        assert chairperson is not None
        assert sample_three_agents is not None
        council = CouncilMeeting(
            council_members=sample_three_agents,
            chairperson=chairperson,
            voting_rounds=1,
            require_consensus=False,
            output_type="str-all-except-first",
        )
        assert council is not None
        result = council.run(sample_task)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) >= 0
        logger.info("CouncilMeeting run test passed")
    except Exception as e:
        logger.error(f"Failed to test CouncilMeeting run: {e}")
        raise


def test_council_meeting_insufficient_members(sample_task):
    try:
        chairperson = create_function_agent("Chairperson")
        single_member = [create_function_agent("Member1")]
        council = CouncilMeeting(
            council_members=single_member,
            chairperson=chairperson,
            voting_rounds=1,
            require_consensus=False,
            output_type="str-all-except-first",
        )
        with pytest.raises(
            ValueError, match="At least two council members"
        ):
            council.run(sample_task)
        logger.info("CouncilMeeting insufficient members test passed")
    except Exception as e:
        logger.error(
            f"Failed to test CouncilMeeting insufficient members: {e}"
        )
        raise


def test_council_meeting_no_chairperson(
    sample_three_agents, sample_task
):
    try:
        council = CouncilMeeting(
            council_members=sample_three_agents,
            chairperson=None,
            voting_rounds=1,
            require_consensus=False,
            output_type="str-all-except-first",
        )
        with pytest.raises(
            ValueError, match="chairperson agent is required"
        ):
            council.run(sample_task)
        logger.info("CouncilMeeting no chairperson test passed")
    except Exception as e:
        logger.error(
            f"Failed to test CouncilMeeting no chairperson: {e}"
        )
        raise


def test_mentorship_session_initialization():
    try:
        mentor = create_function_agent("Mentor")
        assert mentor is not None
        mentee = create_function_agent("Mentee")
        assert mentee is not None
        mentorship = MentorshipSession(
            mentor=mentor,
            mentee=mentee,
            session_count=2,
            include_feedback=True,
            output_type="str-all-except-first",
        )
        assert mentorship is not None
        assert mentorship.mentor is not None
        assert mentorship.mentee is not None
        assert mentorship.session_count == 2
        assert mentorship.include_feedback is True
        logger.info("MentorshipSession initialization test passed")
    except Exception as e:
        logger.error(
            f"Failed to test MentorshipSession initialization: {e}"
        )
        raise


def test_mentorship_session_run(sample_task):
    try:
        mentor = create_function_agent("Mentor")
        assert mentor is not None
        mentee = create_function_agent("Mentee")
        assert mentee is not None
        mentorship = MentorshipSession(
            mentor=mentor,
            mentee=mentee,
            session_count=2,
            include_feedback=True,
            output_type="str-all-except-first",
        )
        assert mentorship is not None
        result = mentorship.run(sample_task)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) >= 0
        logger.info("MentorshipSession run test passed")
    except Exception as e:
        logger.error(f"Failed to test MentorshipSession run: {e}")
        raise


def test_mentorship_session_no_mentor(sample_task):
    try:
        mentee = create_function_agent("Mentee")
        mentorship = MentorshipSession(
            mentor=None,
            mentee=mentee,
            session_count=2,
            include_feedback=True,
            output_type="str-all-except-first",
        )
        with pytest.raises(
            ValueError, match="Both mentor and mentee"
        ):
            mentorship.run(sample_task)
        logger.info("MentorshipSession no mentor test passed")
    except Exception as e:
        logger.error(
            f"Failed to test MentorshipSession no mentor: {e}"
        )
        raise


def test_mentorship_session_no_mentee(sample_task):
    try:
        mentor = create_function_agent("Mentor")
        mentorship = MentorshipSession(
            mentor=mentor,
            mentee=None,
            session_count=2,
            include_feedback=True,
            output_type="str-all-except-first",
        )
        with pytest.raises(
            ValueError, match="Both mentor and mentee"
        ):
            mentorship.run(sample_task)
        logger.info("MentorshipSession no mentee test passed")
    except Exception as e:
        logger.error(
            f"Failed to test MentorshipSession no mentee: {e}"
        )
        raise


def test_negotiation_session_initialization(sample_two_agents):
    try:
        mediator = create_function_agent("Mediator")
        assert mediator is not None
        assert sample_two_agents is not None
        negotiation = NegotiationSession(
            parties=sample_two_agents,
            mediator=mediator,
            negotiation_rounds=3,
            include_concessions=True,
            output_type="str-all-except-first",
        )
        assert negotiation is not None
        assert len(negotiation.parties) == 2
        assert negotiation.mediator is not None
        assert negotiation.negotiation_rounds == 3
        assert negotiation.include_concessions is True
        logger.info("NegotiationSession initialization test passed")
    except Exception as e:
        logger.error(
            f"Failed to test NegotiationSession initialization: {e}"
        )
        raise


def test_negotiation_session_run(sample_two_agents, sample_task):
    try:
        mediator = create_function_agent("Mediator")
        assert mediator is not None
        assert sample_two_agents is not None
        negotiation = NegotiationSession(
            parties=sample_two_agents,
            mediator=mediator,
            negotiation_rounds=2,
            include_concessions=True,
            output_type="str-all-except-first",
        )
        assert negotiation is not None
        result = negotiation.run(sample_task)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) >= 0
        logger.info("NegotiationSession run test passed")
    except Exception as e:
        logger.error(f"Failed to test NegotiationSession run: {e}")
        raise


def test_negotiation_session_insufficient_parties(sample_task):
    try:
        mediator = create_function_agent("Mediator")
        single_party = [create_function_agent("Party1")]
        negotiation = NegotiationSession(
            parties=single_party,
            mediator=mediator,
            negotiation_rounds=2,
            include_concessions=True,
            output_type="str-all-except-first",
        )
        with pytest.raises(ValueError, match="At least two parties"):
            negotiation.run(sample_task)
        logger.info(
            "NegotiationSession insufficient parties test passed"
        )
    except Exception as e:
        logger.error(
            f"Failed to test NegotiationSession insufficient parties: {e}"
        )
        raise


def test_negotiation_session_no_mediator(
    sample_two_agents, sample_task
):
    try:
        negotiation = NegotiationSession(
            parties=sample_two_agents,
            mediator=None,
            negotiation_rounds=2,
            include_concessions=True,
            output_type="str-all-except-first",
        )
        with pytest.raises(
            ValueError, match="mediator agent is required"
        ):
            negotiation.run(sample_task)
        logger.info("NegotiationSession no mediator test passed")
    except Exception as e:
        logger.error(
            f"Failed to test NegotiationSession no mediator: {e}"
        )
        raise


def test_negotiation_session_without_concessions(
    sample_two_agents, sample_task
):
    try:
        mediator = create_function_agent("Mediator")
        assert mediator is not None
        assert sample_two_agents is not None
        negotiation = NegotiationSession(
            parties=sample_two_agents,
            mediator=mediator,
            negotiation_rounds=2,
            include_concessions=False,
            output_type="str-all-except-first",
        )
        assert negotiation is not None
        result = negotiation.run(sample_task)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) >= 0
        logger.info(
            "NegotiationSession without concessions test passed"
        )
    except Exception as e:
        logger.error(
            f"Failed to test NegotiationSession without concessions: {e}"
        )
        raise
