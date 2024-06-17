import pytest
from swarms.structs.rearrange import AgentRearrange


# Mocking the Agent class
class MockAgent:
    def __init__(self, agent_name):
        self.agent_name = agent_name

    def run(self, task):
        return f"Running {task}"


# Test for AgentRearrange class
class TestAgentRearrange:
    @pytest.fixture
    def agent_rearrange(self):
        agents = [MockAgent("agent1"), MockAgent("agent2")]
        return AgentRearrange(agents=agents)

    def test_parse_pattern(self, agent_rearrange):
        assert agent_rearrange.parse_pattern("agent1->agent2") is True
        assert agent_rearrange.parse_pattern("agent3->agent4") is False

    def test_self_find_agent_by_name(self, agent_rearrange):
        assert (
            agent_rearrange.self_find_agent_by_name("agent1").agent_name
            == "agent1"
        )
        assert agent_rearrange.self_find_agent_by_name("agent3") is None

    def test_agent_exists(self, agent_rearrange):
        assert agent_rearrange.agent_exists("agent1") is True
        assert agent_rearrange.agent_exists("agent3") is False

    def test_parse_concurrent_flow(self, agent_rearrange):
        agent_rearrange.parse_concurrent_flow("agent1->agent2")
        assert "agent2" in agent_rearrange.flows["agent1"]

    def test_parse_sequential_flow(self, agent_rearrange):
        agent_rearrange.parse_sequential_flow("agent1", "agent2")
        assert "agent2" in agent_rearrange.flows["agent1"]

    def test_execute_task(self, agent_rearrange):
        assert (
            agent_rearrange.execute_task("agent1", "agent2", "task1", {})
            == "Running task1 (from agent2)"
        )

    def test_process_flows(self, agent_rearrange):
        assert agent_rearrange.process_flows(
            "agent1->agent2", "task1", {}
        ) == ["Running task1"]

    def test_call(self, agent_rearrange):
        assert agent_rearrange(
            pattern="agent1->agent2", default_task="task1"
        ) == ["Running task1"]
