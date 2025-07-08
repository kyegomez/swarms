from swarms import Agent
from swarms.structs.concurrent_workflow import ConcurrentWorkflow


def test_basic_workflow():
    """Test basic workflow initialization and execution"""
    # Create test agents
    agent1 = Agent(
        agent_name="Test-Agent-1",
        system_prompt="You are a test agent 1",
        model_name="claude-3-sonnet-20240229",
        max_loops=1,
    )

    agent2 = Agent(
        agent_name="Test-Agent-2",
        system_prompt="You are a test agent 2",
        model_name="claude-3-sonnet-20240229",
        max_loops=1,
    )

    # Create workflow
    workflow = ConcurrentWorkflow(
        name="test-workflow", agents=[agent1, agent2], max_loops=1
    )

    # Run workflow
    result = workflow.run("Test task")

    # Verify results
    assert len(result) == 2
    assert all(isinstance(r, dict) for r in result)
    assert all("agent" in r and "output" in r for r in result)


def test_dashboard_workflow():
    """Test workflow with dashboard enabled"""
    agent = Agent(
        agent_name="Dashboard-Test-Agent",
        system_prompt="You are a test agent",
        model_name="claude-3-sonnet-20240229",
        max_loops=1,
    )

    workflow = ConcurrentWorkflow(
        name="dashboard-test",
        agents=[agent],
        max_loops=1,
        show_dashboard=True,
    )

    result = workflow.run("Test task")

    assert len(result) == 1
    assert isinstance(result[0], dict)
    assert "agent" in result[0]
    assert "output" in result[0]


def test_multiple_agents():
    """Test workflow with multiple agents"""
    agents = [
        Agent(
            agent_name=f"Agent-{i}",
            system_prompt=f"You are test agent {i}",
            model_name="claude-3-sonnet-20240229",
            max_loops=1,
        )
        for i in range(3)
    ]

    workflow = ConcurrentWorkflow(
        name="multi-agent-test", agents=agents, max_loops=1
    )

    result = workflow.run("Multi-agent test task")

    assert len(result) == 3
    assert all(isinstance(r, dict) for r in result)
    assert all("agent" in r and "output" in r for r in result)


def test_error_handling():
    """Test workflow error handling"""
    # Create an agent that will raise an exception
    agent = Agent(
        agent_name="Error-Agent",
        system_prompt="You are a test agent that will raise an error",
        model_name="invalid-model",  # This will cause an error
        max_loops=1,
    )

    workflow = ConcurrentWorkflow(
        name="error-test", agents=[agent], max_loops=1
    )

    try:
        workflow.run("Test task")
        assert False, "Expected an error but none was raised"
    except Exception as e:
        assert str(e) != ""  # Verify we got an error message


def test_max_loops():
    """Test workflow respects max_loops setting"""
    agent = Agent(
        agent_name="Loop-Test-Agent",
        system_prompt="You are a test agent",
        model_name="claude-3-sonnet-20240229",
        max_loops=2,
    )

    workflow = ConcurrentWorkflow(
        name="loop-test",
        agents=[agent],
        max_loops=1,  # This should override agent's max_loops
    )

    result = workflow.run("Test task")

    assert len(result) == 1
    assert isinstance(result[0], dict)


if __name__ == "__main__":
    test_basic_workflow()
    test_dashboard_workflow()
    test_multiple_agents()
    test_error_handling()
    test_max_loops()
