"""Configuration delivery for ``SelfConsistencyAgent``'s aggregation step."""

from unittest.mock import patch

from swarms.agents import consistency_agent as module
from swarms.agents.consistency_agent import SelfConsistencyAgent


def _run_and_capture_agents(agent):
    """Run ``agent`` with ``Agent`` stubbed; return the kwargs of each construction."""
    constructed = []

    class FakeAgent:
        def __init__(self, *args, **kwargs):
            constructed.append(kwargs)

        def run(self, *args, **kwargs):
            return "sample"

    with patch.object(module, "Agent", FakeAgent):
        agent.run("What is the capital of Australia?")

    return constructed


def _aggregation_kwargs(constructed):
    matches = [
        c
        for c in constructed
        if c.get("agent_name") == "Aggregation-Agent"
    ]
    assert len(matches) == 1, constructed
    return matches[0]


def test_aggregation_uses_the_configured_model():
    agent = SelfConsistencyAgent(
        model_name="claude-sonnet-5", num_samples=2
    )

    kwargs = _aggregation_kwargs(_run_and_capture_agents(agent))

    assert kwargs["model_name"] == "claude-sonnet-5"


def test_aggregation_uses_the_configured_majority_voting_prompt():
    agent = SelfConsistencyAgent(
        majority_voting_prompt="custom voting prompt", num_samples=2
    )

    kwargs = _aggregation_kwargs(_run_and_capture_agents(agent))

    assert kwargs["system_prompt"] == "custom voting prompt"


def test_aggregation_falls_back_to_the_default_prompt_when_none():
    agent = SelfConsistencyAgent(
        majority_voting_prompt=None, num_samples=2
    )

    kwargs = _aggregation_kwargs(_run_and_capture_agents(agent))

    assert kwargs["system_prompt"] == module.majority_voting_prompt


def test_samples_still_use_the_configured_model():
    agent = SelfConsistencyAgent(
        model_name="claude-sonnet-5", num_samples=3
    )

    constructed = _run_and_capture_agents(agent)
    samples = [
        c
        for c in constructed
        if c.get("agent_name") != "Aggregation-Agent"
    ]

    assert len(samples) == 3
    assert {c["model_name"] for c in samples} == {"claude-sonnet-5"}
