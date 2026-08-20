import pytest

from swarms.structs.agent import Agent
from swarms.structs.social_algorithms import (
    AgentNotFoundError,
    SocialAlgorithms,
)


def _agent(name):
    agent = Agent.__new__(Agent)
    agent.agent_name = name
    return agent


def _social_algorithm_with_agents(agent_names):
    social_algorithm = SocialAlgorithms.__new__(SocialAlgorithms)
    social_algorithm.agents = [_agent(name) for name in agent_names]
    social_algorithm.verbose = False
    return social_algorithm


def test_remove_agent_removes_matching_agent_by_name():
    social_algorithm = _social_algorithm_with_agents(
        ["researcher", "critic"]
    )

    social_algorithm.remove_agent("researcher")

    assert [
        agent.agent_name for agent in social_algorithm.agents
    ] == ["critic"]


def test_remove_agent_preserves_remaining_agent_order():
    social_algorithm = _social_algorithm_with_agents(
        ["planner", "researcher", "critic", "writer"]
    )

    social_algorithm.remove_agent("critic")

    assert [
        agent.agent_name for agent in social_algorithm.agents
    ] == ["planner", "researcher", "writer"]


def test_remove_agent_raises_agent_not_found_for_unknown_name():
    social_algorithm = _social_algorithm_with_agents(["researcher"])

    with pytest.raises(AgentNotFoundError):
        social_algorithm.remove_agent("critic")


def test_remove_agent_raises_agent_not_found_for_empty_agent_list():
    social_algorithm = _social_algorithm_with_agents([])

    with pytest.raises(AgentNotFoundError):
        social_algorithm.remove_agent("researcher")
