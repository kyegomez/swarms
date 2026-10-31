from unittest.mock import MagicMock
from swarms.structs.agent import Agent
from swarms.structs.agent_rearrange import AgentRearrange
from swarms.structs.swarm_router import SwarmRouter
from swarms.structs.tree_swarm import TreeAgent
from swarms.structs.advisor_swarm import AdvisorSwarm
from swarms.structs.multi_agent_exec import run_single_agent


def test_agent_call_forwards_positional_args():
    agent = Agent(agent_name="TestAgent", model_name="gpt-4o-mini", max_loops=1)
    agent.run = MagicMock(return_value="mock_result")
    
    # Should forward extra positional arguments without TypeError
    res = agent("test task", None, "extra_pos_arg", key="value")
    
    assert res == "mock_result"
    agent.run.assert_called_once_with("test task", None, "extra_pos_arg", key="value")


def test_agent_run_forwards_positional_args_to_internal_run():
    agent = Agent(agent_name="TestAgent", model_name="gpt-4o-mini", max_loops=1)
    agent._run = MagicMock(return_value="mock_internal")
    
    res = agent.run("test task", "img.png", None, None, None, "extra_pos_arg", key="value")
    
    assert res == "mock_internal"
    agent._run.assert_called_once_with(
        "test task", "img.png", None, None, None, "extra_pos_arg", key="value"
    )


def test_agent_rearrange_run_forwards_positional_args():
    agent1 = Agent(agent_name="A1", model_name="gpt-4o-mini", max_loops=1)
    rearrange = AgentRearrange(agents=[agent1], flow="A1")
    rearrange._run = MagicMock(return_value="mock_rearrange")
    
    res = rearrange.run("rearrange task", "test.png", {"A1": "custom"}, "extra_arg", foo="bar")
    
    assert res == "mock_rearrange"
    rearrange._run.assert_called_once_with(
        "rearrange task", "test.png", {"A1": "custom"}, "extra_arg", foo="bar"
    )


def test_swarm_router_call_forwards_positional_args():
    router = SwarmRouter(name="TestRouter")
    router.run = MagicMock(return_value="mock_router")
    
    res = router("router task", "img.png", None, "extra_arg", param="val")
    
    assert res == "mock_router"
    router.run.assert_called_once_with(
        "router task", "img.png", "extra_arg", param="val"
    )


def test_tree_agent_run_task_forwards_positional_args():
    tree_agent = TreeAgent(agent_name="TreeA", system_prompt="test")
    tree_agent.run = MagicMock(return_value="mock_tree")
    
    res = tree_agent.run_task("tree task", "img.png", "extra_arg", tag="test")
    
    assert res == "mock_tree"
    tree_agent.run.assert_called_once_with("tree task", "img.png", "extra_arg", tag="test")


def test_advisor_swarm_call_forwards_positional_args():
    advisor = AdvisorSwarm(name="TestAdvisor", agents=[])
    advisor.run = MagicMock(return_value="mock_advisor")
    
    res = advisor("advisor task", "extra_arg", mode="fast")
    
    assert res == "mock_advisor"
    advisor.run.assert_called_once_with("advisor task", "extra_arg", mode="fast")


def test_run_single_agent_forwards_positional_args():
    mock_agent = MagicMock(spec=Agent)
    mock_agent.run.return_value = "mock_single"
    
    res = run_single_agent(mock_agent, "single task", "extra_arg", debug=True)
    
    assert res == "mock_single"
    mock_agent.run.assert_called_once_with("single task", "extra_arg", debug=True)
