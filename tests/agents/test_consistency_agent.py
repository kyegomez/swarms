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
    )
