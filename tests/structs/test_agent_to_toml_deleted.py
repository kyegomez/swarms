from swarms.structs.agent import Agent


def test_agent_class_does_not_have_to_toml():
    """Agent class must not define the unused to_toml method."""
    assert not hasattr(Agent, "to_toml"), "Agent should not have to_toml method"


def test_agent_instance_does_not_have_to_toml():
    """Agent instances must not have to_toml attribute."""
    agent = Agent(agent_name="test-agent", max_loops=1)
    assert not hasattr(agent, "to_toml"), "agent instance should not have to_toml method"
