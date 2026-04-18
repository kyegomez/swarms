"""Tests for AdvisorSwarm.

Uses real agents and API calls — no mocks.
"""

import pytest

from swarms.structs.advisor_swarm import AdvisorSwarm
from swarms.structs.agent import Agent


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestAdvisorSwarmInit:
    def test_default_construction(self):
        swarm = AdvisorSwarm()
        assert swarm.name == "AdvisorSwarm"
        assert swarm.executor_model_name == "claude-sonnet-4-6"
        assert swarm.advisor_model_name == "claude-opus-4-6"
        assert swarm.max_advisor_uses == 3
        assert swarm.max_loops == 1
        assert swarm.executor_agent is not None
        assert swarm.advisor_agent is not None
        assert swarm.conversation is not None

    def test_custom_model_names(self):
        swarm = AdvisorSwarm(
            executor_model_name="gpt-4.1-mini",
            advisor_model_name="gpt-4.1",
        )
        assert swarm.executor_model_name == "gpt-4.1-mini"
        assert swarm.advisor_model_name == "gpt-4.1"
        assert swarm.executor_agent.model_name == "gpt-4.1-mini"
        assert swarm.advisor_agent.model_name == "gpt-4.1"

    def test_custom_agents_override(self):
        custom_executor = Agent(
            agent_name="CustomExecutor",
            model_name="gpt-4.1-mini",
            max_loops=1,
        )
        custom_advisor = Agent(
            agent_name="CustomAdvisor",
            model_name="gpt-4.1",
            max_loops=1,
        )
        swarm = AdvisorSwarm(
            executor_agent=custom_executor,
            advisor_agent=custom_advisor,
        )
        assert swarm.executor_agent is custom_executor
        assert swarm.advisor_agent is custom_advisor

    def test_id_generated(self):
        swarm = AdvisorSwarm()
        assert swarm.id is not None
        assert len(swarm.id) > 0

    def test_callable(self):
        swarm = AdvisorSwarm()
        assert callable(swarm)

    def test_zero_advisor_uses_allowed(self):
        """max_advisor_uses=0 means executor runs alone — no advisor."""
        swarm = AdvisorSwarm(max_advisor_uses=0)
        assert swarm.max_advisor_uses == 0


# ---------------------------------------------------------------------------
# Reliability check tests
# ---------------------------------------------------------------------------


class TestReliabilityCheck:
    def test_max_advisor_uses_negative_raises(self):
        with pytest.raises(ValueError, match="max_advisor_uses"):
            AdvisorSwarm(max_advisor_uses=-1)

    def test_max_loops_zero_raises(self):
        with pytest.raises(ValueError, match="max_loops"):
            AdvisorSwarm(max_loops=0)

    def test_empty_executor_model_raises(self):
        with pytest.raises(ValueError, match="executor_model_name"):
            AdvisorSwarm(executor_model_name="")

    def test_empty_advisor_model_raises(self):
        with pytest.raises(ValueError, match="advisor_model_name"):
            AdvisorSwarm(advisor_model_name="")


# ---------------------------------------------------------------------------
# Run validation tests
# ---------------------------------------------------------------------------


class TestRunValidation:
    def test_run_no_task_raises(self):
        swarm = AdvisorSwarm()
        with pytest.raises(ValueError, match="task is required"):
            swarm.run(task=None)

    def test_run_empty_task_raises(self):
        swarm = AdvisorSwarm()
        with pytest.raises(ValueError, match="task is required"):
            swarm.run(task="")


# ---------------------------------------------------------------------------
# Integration tests (require API key)
# ---------------------------------------------------------------------------


class TestAdvisorSwarmExecution:
    """End-to-end tests with real API calls.

    These tests require a valid LLM API key in the environment.
    Skip with: pytest -k 'not Execution'
    """

    def test_single_turn_with_advisor(self):
        swarm = AdvisorSwarm(
            executor_model_name="gpt-4.1-mini",
            advisor_model_name="gpt-4.1-mini",
            max_advisor_uses=1,
            max_loops=1,
        )
        result = swarm.run(task="What is 2 + 2? Answer in one word.")
        assert result is not None

        # Conversation: User, Advisor (1 consultation), Executor (1 turn)
        history = swarm.conversation.to_dict()
        roles = [msg["role"] for msg in history]
        assert "User" in roles
        assert "Advisor" in roles
        assert "Executor" in roles

    def test_executor_only_no_advisor(self):
        """With max_advisor_uses=0, executor runs alone."""
        swarm = AdvisorSwarm(
            executor_model_name="gpt-4.1-mini",
            advisor_model_name="gpt-4.1-mini",
            max_advisor_uses=0,
            max_loops=1,
        )
        result = swarm.run(task="Say hello")
        assert result is not None

        history = swarm.conversation.to_dict()
        roles = [msg["role"] for msg in history]
        assert "Advisor" not in roles
        assert "Executor" in roles

    def test_multi_turn(self):
        """With max_loops=2, executor runs twice."""
        swarm = AdvisorSwarm(
            executor_model_name="gpt-4.1-mini",
            advisor_model_name="gpt-4.1-mini",
            max_advisor_uses=2,
            max_loops=2,
        )
        result = swarm.run(task="Count to 3")
        assert result is not None

        history = swarm.conversation.to_dict()
        executor_entries = [
            msg for msg in history if msg["role"] == "Executor"
        ]
        assert len(executor_entries) == 2

    def test_batched_run(self):
        swarm = AdvisorSwarm(
            executor_model_name="gpt-4.1-mini",
            advisor_model_name="gpt-4.1-mini",
            max_advisor_uses=1,
            max_loops=1,
        )
        results = swarm.batched_run(["Say hi", "Say bye"])
        assert isinstance(results, list)
        assert len(results) == 2

    def test_callable_invocation(self):
        swarm = AdvisorSwarm(
            executor_model_name="gpt-4.1-mini",
            advisor_model_name="gpt-4.1-mini",
            max_advisor_uses=1,
            max_loops=1,
        )
        result = swarm("What is 1 + 1?")
        assert result is not None
