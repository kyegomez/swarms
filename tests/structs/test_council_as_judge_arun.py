import asyncio
from types import SimpleNamespace
import pytest
import swarms.utils.litellm_wrapper as litellm_wrapper
from swarms.structs.council_as_judge import (
    CouncilAsAJudge,
    DimensionEvaluationError,
    EvaluationError,
)


def _mock_completion_setup(monkeypatch):
    requests = []

    def fake_completion(**params):
        requests.append(str(params.get("messages", [])))
        message = SimpleNamespace(
            content="Score: 9/10. Rigorous and accurate analysis.",
            tool_calls=None,
            reasoning_content=None,
        )
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=message, finish_reason="stop")
            ],
            usage=None,
        )

    monkeypatch.setattr(litellm_wrapper, "completion", fake_completion)
    return requests


@pytest.mark.asyncio
async def test_council_as_judge_arun_basic(monkeypatch):
    requests = _mock_completion_setup(monkeypatch)
    council = CouncilAsAJudge(
        model_name="gpt-4o-mini",
        aggregation_model_name="gpt-4o-mini",
    )
    result = await council.arun("Evaluate quantum physics response")
    assert result is not None
    assert len(requests) >= len(council.judge_agents) + 1


@pytest.mark.asyncio
async def test_council_as_judge_arun_max_concurrency(monkeypatch):
    requests = _mock_completion_setup(monkeypatch)
    council = CouncilAsAJudge(
        model_name="gpt-4o-mini",
        aggregation_model_name="gpt-4o-mini",
    )
    result = await council.arun("Evaluate with concurrency cap", max_concurrency=2)
    assert result is not None


@pytest.mark.asyncio
async def test_council_as_judge_arun_native_arun_agents():
    council = CouncilAsAJudge(
        model_name="gpt-4o-mini",
        aggregation_model_name="gpt-4o-mini",
    )
    arun_calls = []

    for dim, agent in council.judge_agents.items():
        async def _make_arun(d, a):
            async def _arun(task=None, messages=None, **kwargs):
                arun_calls.append((d, a.agent_name))
                return f"Evaluation for {d}: pass"
            return _arun
        agent.arun = await _make_arun(dim, agent)

    async def _agg_arun(task=None, messages=None, **kwargs):
        return "Final consensus aggregation report"
    council.aggregator_agent.arun = _agg_arun

    result = await council.arun("Test native arun dispatch")
    assert result is not None
    assert len(arun_calls) == len(council.judge_agents)


@pytest.mark.asyncio
async def test_council_as_judge_arun_dimension_failure():
    council = CouncilAsAJudge(
        model_name="gpt-4o-mini",
        aggregation_model_name="gpt-4o-mini",
    )

    first_dim = next(iter(council.judge_agents.keys()))
    async def _failing_arun(task=None, messages=None, **kwargs):
        raise RuntimeError("API quota exhausted on judge")
    council.judge_agents[first_dim].arun = _failing_arun

    with pytest.raises((DimensionEvaluationError, EvaluationError)):
        await council.arun("Should raise evaluation error")
