from unittest.mock import MagicMock, Mock, patch

import pytest

from swarms.structs.swarm_net import SwarmNet
from swarms.structs.agent import Agent
from swarms.structs.swarm_net import SwarmNetwork


@pytest.fixture
def swarm_network():
    agents = [Agent(id=f"Agent_{i}") for i in range(5)]
    return SwarmNetwork(agents=agents)


def test_swarm_network_init(swarm_network):
    assert isinstance(swarm_network.agents, list)
    assert len(swarm_network.agents) == 5


@patch("swarms.structs.swarm_net.SwarmNetwork.logger")
def test_run(mock_logger, swarm_network):
    swarm_network.run()
    assert (
        mock_logger.info.call_count == 10
    )  # 2 log messages per agent


def test_run_with_mocked_agents(mocker, swarm_network):
    mock_agents = [Mock(spec=Agent) for _ in range(5)]
    mocker.patch.object(swarm_network, "agents", mock_agents)
    swarm_network.run()
    for mock_agent in mock_agents:
        assert mock_agent.run.called


def test_swarm_network_with_no_agents():
    swarm_network = SwarmNetwork(agents=[])
    assert swarm_network.agents == []


def test_swarm_network_add_agent(swarm_network):
    new_agent = Agent(id="Agent_5")
    swarm_network.add_agent(new_agent)
    assert len(swarm_network.agents) == 6
    assert swarm_network.agents[-1] == new_agent


def test_swarm_network_remove_agent(swarm_network):
    agent_to_remove = swarm_network.agents[0]
    swarm_network.remove_agent(agent_to_remove)
    assert len(swarm_network.agents) == 4
    assert agent_to_remove not in swarm_network.agents


@pytest.fixture
def swarmnet():
    swarmnet = SwarmNet()
    agent_mock = MagicMock()
    agent_mock.id = "1"
    swarmnet.agents = [agent_mock]
    return swarmnet


def test_run_agent(swarmnet):
    swarmnet.run_agent("1", "task")
    swarmnet.agents[0].run.assert_called_once_with("task")


def test_run_agent_no_agent(swarmnet):
    with pytest.raises(ValueError, match="No agent found with ID"):
        swarmnet.run_agent("2", "task")


def test_run_many_agents(swarmnet):
    swarmnet.run_many_agents("task")
    swarmnet.agents[0].run.assert_called_once_with("task")


def test_list_agents(swarmnet):
    swarmnet.list_agents()
    assert swarmnet.agents[0].id == "1"


def test_get_agent(swarmnet):
    agent = swarmnet.get_agent("1")
    assert agent.id == "1"


def test_get_agent_no_agent(swarmnet):
    with pytest.raises(ValueError, match="No agent found with ID"):
        swarmnet.get_agent("2")
