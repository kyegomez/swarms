import gc
import pytest
from unittest.mock import MagicMock

from swarms.agents.consistency_agent import SelfConsistencyAgent


class TestSelfConsistencyAgentExecutorReuse:
    """Tests verifying thread pool executor reuse, context management, and cleanup in SelfConsistencyAgent."""

    def test_self_consistency_agent_reuses_executor_across_runs(self, monkeypatch):
        agent = SelfConsistencyAgent(
            name="Test-Consistency-Agent",
            model_name="gpt-4o-mini",
            num_samples=3,
        )

        dummy_reasoning_agent = MagicMock()
        dummy_reasoning_agent.run.return_value = "Answer 42"
        monkeypatch.setattr(
            agent, "_create_reasoning_agent", lambda: dummy_reasoning_agent
        )
        monkeypatch.setattr(
            "swarms.agents.consistency_agent.aggregation_agent",
            lambda responses, **kw: "Aggregated: Answer 42",
        )

        res1 = agent.run("Problem 1")
        assert res1 is not None
        exec1 = getattr(agent, "_executor", None)
        assert exec1 is not None

        res2 = agent.run("Problem 2")
        assert res2 is not None
        exec2 = getattr(agent, "_executor", None)
        assert exec1 is exec2
        assert not exec1._shutdown

    def test_self_consistency_agent_context_manager_and_close(self):
        with SelfConsistencyAgent(
            name="Test-Consistency-Agent",
            model_name="gpt-4o-mini",
            num_samples=2,
        ) as agent:
            executor = agent._get_executor()
            assert executor is not None
            assert not executor._shutdown

        assert agent._executor is None
        assert executor._shutdown

    def test_self_consistency_agent_weakref_finalizer(self):
        agent = SelfConsistencyAgent(
            name="Test-Consistency-Agent",
            model_name="gpt-4o-mini",
            num_samples=2,
        )
        executor = agent._get_executor()
        assert not executor._shutdown

        del agent
        gc.collect()

        assert executor._shutdown

    def test_self_consistency_agent_resizes_when_larger_workers_requested(self):
        agent = SelfConsistencyAgent(
            name="Test-Consistency-Agent",
            model_name="gpt-4o-mini",
            num_samples=2,
        )
        exec1 = agent._get_executor(max_workers=2)
        assert getattr(exec1, "_max_workers", None) == 2

        exec2 = agent._get_executor(max_workers=6)
        assert exec1 is not exec2
        assert exec1._shutdown
        assert getattr(exec2, "_max_workers", None) == 6

        agent.close()
        assert exec2._shutdown
