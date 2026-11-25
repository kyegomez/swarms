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


@pytest.mark.parametrize(
    "preset_agents, aggregator_agent, message",
    [
        (True, object(), "set preset_agents=False"),
        (False, None, "needs an aggregator_agent"),
    ],
    ids=["aggregator-with-presets", "no-aggregator-without-presets"],
)
def test_preset_agents_and_aggregator_agent_must_agree(
    preset_agents, aggregator_agent, message
):
    with pytest.raises(ValueError, match=message):
        SelfConsistencyAgent(
            preset_agents=preset_agents,
            aggregator_agent=aggregator_agent,
        )


class _CallerAggregator:
    def __init__(self):
        self.tasks = []
        self.resets = 0
        self.usage = _totals(5)

    def reset(self):
        self.resets += 1

    def run(self, task, *args, **kwargs):
        self.tasks.append(task)
        self.usage = _totals(5 + len(self.tasks))
        return "voted"


def test_caller_aggregator_votes_and_counts_only_its_own_calls():
    aggregator = _CallerAggregator()
    agent = SelfConsistencyAgent(
        num_samples=2,
        preset_agents=False,
        aggregator_agent=aggregator,
        output_type="final",
    )

    with patch("swarms.agents.consistency_agent.Agent", _UsageAgent):
        assert agent.run("task") == "voted"
        agent.run("task")

    assert aggregator.resets == 2
    assert all(task.count("sample") == 2 for task in aggregator.tasks)
    assert agent.usage == _totals(2 * 2 + 2)


@pytest.mark.parametrize(
    "sample, accepted",
    [
        ("2 + 2 = 4.", True),
        ("The answer is 4", True),
        ("2 + 2 = 14", False),
        ("The answer is 42.", False),
        ("I get 0.4", False),
        ("I get 4.5", False),
        ("The answer is -4.", False),
        ("The total is 4,000", False),
        ("The ratio is 3/4", False),
    ],
)
def test_eval_accepts_only_the_whole_answer(sample, accepted):
    class _SampleAgent(_UsageAgent):
        def run(self, task, *args, **kwargs):
            return sample

    agent = SelfConsistencyAgent(
        num_samples=2, eval=True, output_type="final"
    )

    with patch("swarms.agents.consistency_agent.Agent", _SampleAgent):
        result = agent.run("What is 2 + 2?", answer="4")

    assert (result is not None) is accepted
