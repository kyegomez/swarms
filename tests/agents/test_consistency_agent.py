from unittest.mock import patch

import pytest

from swarms.agents.consistency_agent import (
    SelfConsistencyAgent,
    majority_voting_prompt,
)


@pytest.mark.parametrize(
    "configured, expected",
    [("custom", "custom"), (None, majority_voting_prompt)],
    ids=["custom", "none"],
)
def test_run_aggregates_on_configured_model_and_prompt(
    configured, expected
):
    agent = SelfConsistencyAgent(
        model_name="claude-sonnet-5",
        majority_voting_prompt=configured,
        num_samples=2,
    )

    with (
        patch.object(agent, "_create_reasoning_agent") as create,
        patch(
            "swarms.agents.consistency_agent.aggregation_agent",
            return_value="final",
        ) as aggregate,
    ):
        create.return_value.run.return_value = "sample"
        agent.run("task")

    aggregate.assert_called_once_with(
        ["sample", "sample"],
        prompt=expected,
        model_name="claude-sonnet-5",
        on_usage=agent._add_usage,
    )


class _UsageAgent:
    def __init__(self, *args, **kwargs):
        self.usage = {
            "input_tokens": 10,
            "output_tokens": 2,
            "cached_tokens": 0,
            "reasoning_tokens": 1,
            "total_tokens": 12,
        }

    def run(self, task, *args, **kwargs):
        return "sample"


def _totals(calls):
    return {
        "input_tokens": 10 * calls,
        "output_tokens": 2 * calls,
        "cached_tokens": 0,
        "reasoning_tokens": calls,
        "total_tokens": 12 * calls,
    }


def test_usage_sums_samples_and_aggregator_across_runs():
    agent = SelfConsistencyAgent(num_samples=3)
    assert agent.usage == _totals(0)

    with patch("swarms.agents.consistency_agent.Agent", _UsageAgent):
        agent.run("task")
        assert agent.usage == _totals(4)
        agent.batched_run(["a", "b"])

    assert agent.usage == _totals(12)


def test_usage_counts_samples_when_eval_returns_early():
    agent = SelfConsistencyAgent(num_samples=2, eval=True)

    with patch("swarms.agents.consistency_agent.Agent", _UsageAgent):
        assert agent.run("task", answer="missing") is None

    assert agent.usage == _totals(2)
