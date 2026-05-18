import os
import pytest

from swarms import Agent
from swarms.structs.concurrent_workflow import ConcurrentWorkflow
from swarms.utils.workspace_utils import get_workspace_dir


def test_concurrent_workflow_basic_execution():
    """Test basic ConcurrentWorkflow execution with multiple agents"""
    # Create specialized agents for different perspectives
    research_agent = Agent(
        agent_name="Research-Analyst",
        agent_description="Agent specializing in research and data collection",
        model_name="gpt-5.4",
        verbose=False,
        print_on=False,
        max_loops=1,
    )

    strategy_agent = Agent(
        agent_name="Strategy-Consultant",
        agent_description="Agent specializing in strategic planning and analysis",
        model_name="gpt-5.4",
        verbose=False,
        print_on=False,
        max_loops=1,
    )

    risk_agent = Agent(
        agent_name="Risk-Assessment-Specialist",
        agent_description="Agent specializing in risk analysis and mitigation",
        model_name="gpt-5.4",
        verbose=False,
        print_on=False,
        max_loops=1,
    )

    # Create workflow with multiple agents
    workflow = ConcurrentWorkflow(
        name="Multi-Perspective-Analysis-Workflow",
        description="Concurrent analysis from research, strategy, and risk perspectives",
        agents=[research_agent, strategy_agent, risk_agent],
        max_loops=1,
    )

    # Run workflow
    result = workflow.run(
        "Analyze the potential impact of quantum computing on cybersecurity"
    )

    # Verify results - ConcurrentWorkflow with default output_type returns conversation history
    assert result is not None
    assert isinstance(result, list)
    # With default output_type="dict-all-except-first", we get conversation_history[2:]
    # So we should have at least some results, but exact count depends on successful agent runs
    assert len(result) >= 1  # At least one agent should succeed
    for r in result:
        assert isinstance(r, dict)
        assert "role" in r  # Agent name is stored in 'role' field
        assert (
            "content" in r
        )  # Agent output is stored in 'content' field


def test_concurrent_workflow_with_dashboard():
    """Test ConcurrentWorkflow with dashboard visualization"""
    # Create agents with different expertise
    market_agent = Agent(
        agent_name="Market-Analyst",
        agent_description="Agent for market analysis and trends",
        model_name="gpt-5.4",
        verbose=False,
        print_on=False,
        max_loops=1,
    )

    financial_agent = Agent(
        agent_name="Financial-Expert",
        agent_description="Agent for financial analysis and forecasting",
        model_name="gpt-5.4",
        verbose=False,
        print_on=False,
        max_loops=1,
    )

    technology_agent = Agent(
        agent_name="Technology-Specialist",
        agent_description="Agent for technology assessment and innovation",
        model_name="gpt-5.4",
        verbose=False,
        print_on=False,
        max_loops=1,
    )

    workflow = ConcurrentWorkflow(
        name="Dashboard-Analysis-Workflow",
        description="Concurrent analysis with real-time dashboard monitoring",
        agents=[market_agent, financial_agent, technology_agent],
        max_loops=1,
        show_dashboard=True,
    )

    result = workflow.run(
        "Evaluate investment opportunities in renewable energy sector"
    )

    assert result is not None
    assert isinstance(result, list)
    # With default output_type="dict-all-except-first", we get conversation_history[2:]
    assert len(result) >= 1  # At least one agent should succeed
    for r in result:
        assert isinstance(r, dict)
        assert "role" in r  # Agent name is stored in 'role' field
        assert (
            "content" in r
        )  # Agent output is stored in 'content' field


def test_concurrent_workflow_batched_execution():
    """Test batched execution of multiple tasks"""
    # Create agents for comprehensive analysis
    agents = [
        Agent(
            agent_name=f"Analysis-Agent-{i+1}",
            agent_description=f"Agent {i+1} for comprehensive business analysis",
            model_name="gpt-5.4",
            verbose=False,
            print_on=False,
            max_loops=1,
        )
        for i in range(4)
    ]

    workflow = ConcurrentWorkflow(
        name="Batched-Analysis-Workflow",
        description="Workflow for processing multiple analysis tasks",
        agents=agents,
        max_loops=1,
    )

    # Test batched execution
    tasks = [
        "Analyze market trends in AI adoption",
        "Evaluate competitive landscape in cloud computing",
        "Assess regulatory impacts on fintech",
        "Review supply chain vulnerabilities in manufacturing",
    ]

    results = workflow.batch_run(tasks)

    assert results is not None
    assert isinstance(results, list)
    assert len(results) == 4
    # Each result should be a list of agent outputs
    for result in results:
        assert result is not None
        assert isinstance(result, list)


def test_concurrent_workflow_error_handling():
    """Test ConcurrentWorkflow error handling and validation"""
    # Test with empty agents list
    try:
        ConcurrentWorkflow(agents=[])
        assert (
            False
        ), "Should have raised ValueError for empty agents list"
    except ValueError as e:
        assert "No agents provided" in str(e)

    # Test with None agents
    try:
        ConcurrentWorkflow(agents=None)
        assert False, "Should have raised ValueError for None agents"
    except ValueError as e:
        assert "No agents provided" in str(e)
        assert str(e) != ""  # Verify we got an error message


def test_concurrent_workflow_max_loops_configuration():
    """Test ConcurrentWorkflow max_loops configuration"""
    agent1 = Agent(
        agent_name="Loop-Test-Agent-1",
        agent_description="First agent for loop testing",
        model_name="gpt-5.4",
        verbose=False,
        print_on=False,
        max_loops=2,
    )

    agent2 = Agent(
        agent_name="Loop-Test-Agent-2",
        agent_description="Second agent for loop testing",
        model_name="gpt-5.4",
        verbose=False,
        print_on=False,
        max_loops=3,
    )

    workflow = ConcurrentWorkflow(
        name="Loop-Configuration-Test",
        description="Testing max_loops configuration",
        agents=[agent1, agent2],
        max_loops=1,  # This should override agent's max_loops
    )

    result = workflow.run("Test workflow loop configuration")

    assert result is not None
    assert isinstance(result, list)
    # With default output_type="dict-all-except-first", we get conversation_history[2:]
    assert len(result) >= 1  # At least one agent should succeed
    for r in result:
        assert isinstance(r, dict)
        assert "role" in r  # Agent name is stored in 'role' field
        assert (
            "content" in r
        )  # Agent output is stored in 'content' field


def test_concurrent_workflow_different_output_types():
    """Test ConcurrentWorkflow with different output types"""
    # Create agents with diverse perspectives
    technical_agent = Agent(
        agent_name="Technical-Analyst",
        agent_description="Agent for technical analysis",
        model_name="gpt-5.4",
        verbose=False,
        print_on=False,
        max_loops=1,
    )

    business_agent = Agent(
        agent_name="Business-Strategist",
        agent_description="Agent for business strategy",
        model_name="gpt-5.4",
        verbose=False,
        print_on=False,
        max_loops=1,
    )

    legal_agent = Agent(
        agent_name="Legal-Expert",
        agent_description="Agent for legal compliance analysis",
        model_name="gpt-5.4",
        verbose=False,
        print_on=False,
        max_loops=1,
    )

    # Test different output types
    for output_type in ["dict", "dict-all-except-first"]:
        workflow = ConcurrentWorkflow(
            name=f"Output-Type-Test-{output_type}",
            description=f"Testing output type: {output_type}",
            agents=[technical_agent, business_agent, legal_agent],
            max_loops=1,
            output_type=output_type,
        )

        result = workflow.run("Evaluate AI implementation strategy")
        assert result is not None
        # The result structure depends on output_type, just ensure it's not None


def test_concurrent_workflow_real_world_scenario():
    """Test ConcurrentWorkflow in a realistic business scenario"""
    # Create agents representing different departments
    marketing_agent = Agent(
        agent_name="Marketing-Director",
        agent_description="Senior marketing director with 15 years experience",
        model_name="gpt-5.4",
        verbose=False,
        print_on=False,
        max_loops=1,
    )

    product_agent = Agent(
        agent_name="Product-Manager",
        agent_description="Product manager specializing in AI/ML products",
        model_name="gpt-5.4",
        verbose=False,
        print_on=False,
        max_loops=1,
    )

    engineering_agent = Agent(
        agent_name="Lead-Engineer",
        agent_description="Senior software engineer and technical architect",
        model_name="gpt-5.4",
        verbose=False,
        print_on=False,
        max_loops=1,
    )

    sales_agent = Agent(
        agent_name="Sales-Executive",
        agent_description="Enterprise sales executive with tech background",
        model_name="gpt-5.4",
        verbose=False,
        print_on=False,
        max_loops=1,
    )

    workflow = ConcurrentWorkflow(
        name="Product-Launch-Review-Workflow",
        description="Cross-functional team reviewing new AI product launch strategy",
        agents=[
            marketing_agent,
            product_agent,
            engineering_agent,
            sales_agent,
        ],
        max_loops=1,
    )

    # Test with a realistic business scenario
    result = workflow.run(
        "Review and provide recommendations for our new AI-powered analytics platform launch. "
        "Consider market positioning, technical feasibility, competitive landscape, and sales strategy."
    )

    assert result is not None
    assert isinstance(result, list)
    # With default output_type="dict-all-except-first", we get conversation_history[2:]
    assert len(result) >= 1  # At least one agent should succeed
    for r in result:
        assert isinstance(r, dict)
        assert "role" in r  # Agent name is stored in 'role' field
        assert (
            "content" in r
        )  # Agent output is stored in 'content' field


def test_concurrent_workflow_team_collaboration():
    """Test ConcurrentWorkflow with team collaboration features"""
    # Create agents that would naturally collaborate
    data_scientist = Agent(
        agent_name="Data-Scientist",
        agent_description="ML engineer and data scientist",
        model_name="gpt-5.4",
        verbose=False,
        print_on=False,
        max_loops=1,
    )

    ux_designer = Agent(
        agent_name="UX-Designer",
        agent_description="User experience designer and researcher",
        model_name="gpt-5.4",
        verbose=False,
        print_on=False,
        max_loops=1,
    )

    product_owner = Agent(
        agent_name="Product-Owner",
        agent_description="Product owner with business and technical background",
        model_name="gpt-5.4",
        verbose=False,
        print_on=False,
        max_loops=1,
    )

    qa_engineer = Agent(
        agent_name="QA-Engineer",
        agent_description="Quality assurance engineer and testing specialist",
        model_name="gpt-5.4",
        verbose=False,
        print_on=False,
        max_loops=1,
    )

    workflow = ConcurrentWorkflow(
        name="Cross-Functional-Development-Workflow",
        description="Cross-functional team collaborating on feature development",
        agents=[
            data_scientist,
            ux_designer,
            product_owner,
            qa_engineer,
        ],
        max_loops=1,
    )

    result = workflow.run(
        "Design and plan a new recommendation system for our e-commerce platform. "
        "Each team member should provide their perspective on implementation, user experience, "
        "business value, and quality assurance considerations."
    )

    assert result is not None
    assert isinstance(result, list)
    # With default output_type="dict-all-except-first", we get conversation_history[2:]
    assert len(result) >= 1  # At least one agent should succeed
    for r in result:
        assert isinstance(r, dict)
        assert "role" in r  # Agent name is stored in 'role' field
        assert (
            "content" in r
        )  # Agent output is stored in 'content' field


def test_concurrent_workflow_autosave_creates_workspace_dir(
    monkeypatch, tmp_path
):
    """Test that ConcurrentWorkflow with autosave=True creates a workspace directory."""
    get_workspace_dir.cache_clear()
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path))

    agent1 = Agent(
        agent_name="Autosave-Concurrent-1",
        agent_description="Agent for autosave test",
        model_name="gpt-5.4",
        max_loops=1,
        verbose=False,
        print_on=False,
    )
    agent2 = Agent(
        agent_name="Autosave-Concurrent-2",
        agent_description="Agent for autosave test",
        model_name="gpt-5.4",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    workflow = ConcurrentWorkflow(
        name="Autosave-Concurrent-Workflow",
        agents=[agent1, agent2],
        max_loops=1,
        autosave=True,
        verbose=False,
    )

    assert workflow.autosave is True
    assert workflow.swarm_workspace_dir is not None
    assert os.path.isdir(workflow.swarm_workspace_dir)
    assert "ConcurrentWorkflow" in workflow.swarm_workspace_dir
    assert (
        "Autosave-Concurrent-Workflow" in workflow.swarm_workspace_dir
    )

    get_workspace_dir.cache_clear()


def test_concurrent_workflow_autosave_saves_conversation_after_run(
    monkeypatch, tmp_path
):
    """Test that ConcurrentWorkflow saves conversation_history.json after run when autosave=True."""
    get_workspace_dir.cache_clear()
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path))

    agent1 = Agent(
        agent_name="Autosave-Run-Concurrent-1",
        agent_description="Agent for autosave run test",
        model_name="gpt-5.4",
        max_loops=1,
        verbose=False,
        print_on=False,
    )
    agent2 = Agent(
        agent_name="Autosave-Run-Concurrent-2",
        agent_description="Agent for autosave run test",
        model_name="gpt-5.4",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    workflow = ConcurrentWorkflow(
        name="Autosave-Run-Concurrent-Workflow",
        agents=[agent1, agent2],
        max_loops=1,
        autosave=True,
        verbose=False,
    )

    result = workflow.run("Say hello in one short sentence.")
    assert result is not None

    conversation_path = os.path.join(
        workflow.swarm_workspace_dir, "conversation_history.json"
    )
    assert os.path.isfile(
        conversation_path
    ), f"Expected conversation_history.json at {conversation_path}"

    get_workspace_dir.cache_clear()


class _MockAgent:
    """Minimal stand-in that looks like an Agent to ConcurrentWorkflow."""

    def __init__(self, name: str, response: str = "ok"):
        self.agent_name = name
        self._response = response
        self.print_on = True

    def run(self, task=None, img=None, imgs=None, **kwargs):
        return self._response

    def cleanup(self):
        pass


class _FailingAgent(_MockAgent):
    def run(self, task=None, img=None, imgs=None, **kwargs):
        raise RuntimeError(f"{self.agent_name} exploded")


# ---------------------------------------------------------------------------
# on_error tests (no real LLM calls)
# ---------------------------------------------------------------------------


def test_concurrent_workflow_partial_results_on_agent_failure():
    """One failing agent must not discard outputs from passing siblings."""
    good1 = _MockAgent("Good-1", "result-1")
    good2 = _MockAgent("Good-2", "result-2")
    bad = _FailingAgent("Bad")

    wf = ConcurrentWorkflow(
        agents=[good1, good2, bad],
        on_error="store",
        autosave=False,
    )
    result = wf.run("test task")

    # Must not raise — partial results are returned
    assert result is not None

    # Inspect the raw conversation history so no formatter slicing can hide entries.
    history = wf.conversation.conversation_history
    roles = [entry.get("role", "") for entry in history]
    contents = [entry.get("content", "") for entry in history]

    # Both passing agents must appear
    assert any("Good-1" in r for r in roles), "Good-1 output missing"
    assert any("Good-2" in r for r in roles), "Good-2 output missing"

    # The failing agent must appear under the "(failed)" marker
    assert any(
        "Bad" in r and "failed" in r for r in roles
    ), "Failed-agent error entry missing"

    # Error content stored
    assert any(
        "exploded" in c for c in contents
    ), "Error message not stored in conversation"


def test_concurrent_workflow_on_error_raise_propagates():
    """on_error='raise' must let the exception escape."""
    good = _MockAgent("Good", "ok")
    bad = _FailingAgent("Bad")

    wf = ConcurrentWorkflow(
        agents=[good, bad],
        on_error="raise",
        autosave=False,
    )
    with pytest.raises(RuntimeError, match="exploded"):
        wf.run("test task")


def test_concurrent_workflow_invalid_on_error_raises_value_error():
    """Unsupported on_error value must raise ValueError at construction time."""
    with pytest.raises(ValueError, match="on_error"):
        ConcurrentWorkflow(
            agents=[_MockAgent("A"), _MockAgent("B")],
            on_error="ignore",  # invalid
            autosave=False,
        )


# ---------------------------------------------------------------------------
# Dashboard-path tests (show_dashboard=True, renderer stubbed out)
# ---------------------------------------------------------------------------


def test_concurrent_workflow_dashboard_partial_results_on_agent_failure(
    monkeypatch,
):
    """Dashboard path: failing agent must not discard passing siblings."""
    good1 = _MockAgent("Good-1", "result-1")
    good2 = _MockAgent("Good-2", "result-2")
    bad = _FailingAgent("Bad")

    # Stub out all dashboard rendering so no terminal UI code runs.
    monkeypatch.setattr(
        "swarms.structs.concurrent_workflow.formatter",
        _NullFormatter(),
        raising=False,
    )

    wf = ConcurrentWorkflow(
        agents=[good1, good2, bad],
        on_error="store",
        show_dashboard=True,
        autosave=False,
    )
    result = wf.run("test task")

    assert result is not None

    history = wf.conversation.conversation_history
    roles = [entry.get("role", "") for entry in history]
    contents = [entry.get("content", "") for entry in history]

    assert any("Good-1" in r for r in roles), "Good-1 output missing"
    assert any("Good-2" in r for r in roles), "Good-2 output missing"
    assert any(
        "Bad" in r and "failed" in r for r in roles
    ), "Failed-agent error entry missing"
    assert any(
        "exploded" in c for c in contents
    ), "Error message not stored"


def test_concurrent_workflow_dashboard_on_error_raise_propagates(
    monkeypatch,
):
    """Dashboard path: on_error='raise' must propagate the exception."""
    good = _MockAgent("Good", "ok")
    bad = _FailingAgent("Bad")

    monkeypatch.setattr(
        "swarms.structs.concurrent_workflow.formatter",
        _NullFormatter(),
        raising=False,
    )

    wf = ConcurrentWorkflow(
        agents=[good, bad],
        on_error="raise",
        show_dashboard=True,
        autosave=False,
    )
    with pytest.raises(RuntimeError, match="exploded"):
        wf.run("test task")


class _NullFormatter:
    """Drop-in for ``formatter`` that silently ignores all dashboard calls."""

    def print_agent_dashboard(self, *a, **kw):
        pass

    def stop_dashboard(self, *a, **kw):
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
