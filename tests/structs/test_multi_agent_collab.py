import json
import os
import pytest
from unittest.mock import Mock
from swarms.structs import Agent
from swarms.models import OpenAIChat
from swarms.structs.multi_agent_collab import (
    MultiAgentCollaboration,
)

# Sample agents for testing
agent1 = Agent(llm=OpenAIChat(), max_loops=2)
agent2 = Agent(llm=OpenAIChat(), max_loops=2)
agents = [agent1, agent2]


@pytest.fixture
def collaboration():
    return MultiAgentCollaboration(agents)


def test_collaboration_initialization(collaboration):
    assert len(collaboration.agents) == 2
    assert callable(collaboration.select_next_speaker)
    assert collaboration.max_iters == 10
    assert collaboration.results == []
    assert collaboration.logging is True


def test_reset(collaboration):
    collaboration.reset()
    for agent in collaboration.agents:
        assert agent.step == 0


def test_inject(collaboration):
    collaboration.inject("TestName", "TestMessage")
    for agent in collaboration.agents:
        assert "TestName" in agent.history[-1]
        assert "TestMessage" in agent.history[-1]


def test_inject_agent(collaboration):
    agent3 = Agent(llm=OpenAIChat(), max_loops=2)
    collaboration.inject_agent(agent3)
    assert len(collaboration.agents) == 3
    assert agent3 in collaboration.agents


def test_step(collaboration):
    collaboration.step()
    for agent in collaboration.agents:
        assert agent.step == 1


def test_ask_for_bid(collaboration):
    agent = Mock()
    agent.bid.return_value = "<5>"
    bid = collaboration.ask_for_bid(agent)
    assert bid == 5


def test_select_next_speaker(collaboration):
    collaboration.select_next_speaker = Mock(return_value=0)
    idx = collaboration.select_next_speaker(1, collaboration.agents)
    assert idx == 0


def test_run(collaboration):
    collaboration.run()
    for agent in collaboration.agents:
        assert agent.step == collaboration.max_iters


def test_format_results(collaboration):
    collaboration.results = [
        {"agent": "Agent1", "response": "Response1"}
    ]
    formatted_results = collaboration.format_results(
        collaboration.results
    )
    assert "Agent1 responded: Response1" in formatted_results


def test_save_and_load(collaboration):
    collaboration.save()
    loaded_state = collaboration.load()
    assert loaded_state["_step"] == collaboration._step
    assert loaded_state["results"] == collaboration.results


def test_performance(collaboration):
    performance_data = collaboration.performance()
    for agent in collaboration.agents:
        assert agent.name in performance_data
        assert "metrics" in performance_data[agent.name]


def test_set_interaction_rules(collaboration):
    rules = {"rule1": "action1", "rule2": "action2"}
    collaboration.set_interaction_rules(rules)
    assert hasattr(collaboration, "interaction_rules")
    assert collaboration.interaction_rules == rules


def test_repr(collaboration):
    repr_str = repr(collaboration)
    assert isinstance(repr_str, str)
    assert "MultiAgentCollaboration" in repr_str


def test_load(collaboration):
    state = {
        "step": 5,
        "results": [{"agent": "Agent1", "response": "Response1"}],
    }
    with open(collaboration.saved_file_path_name, "w") as file:
        json.dump(state, file)

    loaded_state = collaboration.load()
    assert loaded_state["_step"] == state["step"]
    assert loaded_state["results"] == state["results"]


def test_save(collaboration, tmp_path):
    collaboration.saved_file_path_name = tmp_path / "test_save.json"
    collaboration.save()

    with open(collaboration.saved_file_path_name, "r") as file:
        saved_data = json.load(file)

    assert saved_data["_step"] == collaboration._step
    assert saved_data["results"] == collaboration.results


# Add more tests here...

# Add more parameterized tests for different scenarios...


# Example of exception testing
def test_exception_handling(collaboration):
    agent = Mock()
    agent.bid.side_effect = ValueError("Invalid bid")
    with pytest.raises(ValueError):
        collaboration.ask_for_bid(agent)


# Add more exception testing...


# Example of environment variable testing (if applicable)
@pytest.mark.parametrize("env_var", ["ENV_VAR_1", "ENV_VAR_2"])
def test_environment_variables(collaboration, monkeypatch, env_var):
    monkeypatch.setenv(env_var, "test_value")
    assert os.getenv(env_var) == "test_value"
