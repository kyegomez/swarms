"""Tests for AutoAgentBuilder's handling of agent_kwargs.

agent_kwargs is documented as extra arguments for the *generated* agents, and
as ignoring keys the builder fills in itself. Both properties were broken by a
keyword collision that raised TypeError before any LLM call, so every example
passing agent_kwargs={"max_loops": 1} failed on the first line of run().

No network: the collision happens while constructing the Agent objects.
"""

import pytest

from swarms.structs.auto_agent_builder import AutoAgentBuilder


def test_agent_kwargs_reaches_generated_agents_without_colliding():
    """max_loops in agent_kwargs must not clash with the builder's own."""
    builder = AutoAgentBuilder(agent_kwargs={"max_loops": 3})

    # The builder agent is constructed before the roster call; this raised
    # "got multiple values for keyword argument 'max_loops'".
    assert builder._builder_agent().max_loops == 1

    assert builder.agent_kwargs["max_loops"] == 3


def test_generated_fields_are_dropped_from_agent_kwargs():
    """Keys the roster supplies per agent are ignored, as documented."""
    builder = AutoAgentBuilder(
        agent_kwargs={
            "model_name": "gpt-5.4-mini",
            "agent_name": "override",
            "streaming_on": True,
        }
    )

    assert "model_name" not in builder.agent_kwargs
    assert "agent_name" not in builder.agent_kwargs
    assert builder.agent_kwargs["streaming_on"] is True


def test_roster_count_is_validated():
    """num_agents and max_agents below 1 are rejected at construction."""
    with pytest.raises(ValueError):
        AutoAgentBuilder(max_agents=0)
    with pytest.raises(ValueError):
        AutoAgentBuilder(num_agents=0)
