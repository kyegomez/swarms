import os
import pytest

from swarms import Agent
from swarms.structs.hiearchical_swarm import HierarchicalSwarm
from swarms.utils.workspace_utils import get_workspace_dir


def test_hierarchical_swarm_basic_initialization():
    """Test basic HierarchicalSwarm initialization"""
    # Create worker agents
    research_agent = Agent(
        agent_name="Research-Specialist",
        agent_description="Specialist in research and data collection",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    analysis_agent = Agent(
        agent_name="Analysis-Expert",
        agent_description="Expert in data analysis and insights",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    implementation_agent = Agent(
        agent_name="Implementation-Manager",
        agent_description="Manager for implementation and execution",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    # Create swarm with agents
    swarm = HierarchicalSwarm(
        name="Research-Analysis-Implementation-Swarm",
        description="Hierarchical swarm for comprehensive project execution",
        agents=[research_agent, analysis_agent, implementation_agent],
        max_loops=1,
    )

    # Verify initialization
    assert swarm.name == "Research-Analysis-Implementation-Swarm"
    assert (
        swarm.description
        == "Hierarchical swarm for comprehensive project execution"
    )
    assert len(swarm.agents) == 3
    assert swarm.max_loops == 1
    assert swarm.director is not None


def test_hierarchical_swarm_with_director():
    """Test HierarchicalSwarm with custom director"""
    # Create a custom director
    director = Agent(
        agent_name="Project-Director",
        agent_description="Senior project director with extensive experience",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    # Create worker agents
    developer = Agent(
        agent_name="Senior-Developer",
        agent_description="Senior software developer",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    tester = Agent(
        agent_name="QA-Lead",
        agent_description="Quality assurance lead",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    # Create swarm with custom director
    swarm = HierarchicalSwarm(
        name="Software-Development-Swarm",
        description="Hierarchical swarm for software development projects",
        director=director,
        agents=[developer, tester],
        max_loops=2,
    )

    assert swarm.director == director
    assert len(swarm.agents) == 2
    assert swarm.max_loops == 2


def test_hierarchical_swarm_execution():
    """Test HierarchicalSwarm execution with multiple agents"""
    # Create specialized agents
    market_researcher = Agent(
        agent_name="Market-Researcher",
        agent_description="Market research specialist",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    product_strategist = Agent(
        agent_name="Product-Strategist",
        agent_description="Product strategy and planning expert",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    technical_architect = Agent(
        agent_name="Technical-Architect",
        agent_description="Technical architecture and design specialist",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    risk_analyst = Agent(
        agent_name="Risk-Analyst",
        agent_description="Risk assessment and mitigation specialist",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    # Create hierarchical swarm
    swarm = HierarchicalSwarm(
        name="Product-Development-Swarm",
        description="Comprehensive product development hierarchical swarm",
        agents=[
            market_researcher,
            product_strategist,
            technical_architect,
            risk_analyst,
        ],
        max_loops=1,
        verbose=True,
    )

    # Execute swarm
    result = swarm.run(
        "Develop a comprehensive strategy for a new AI-powered healthcare platform"
    )

    # Verify result structure
    assert result is not None
    # HierarchicalSwarm returns a SwarmSpec or conversation history, just ensure it's not None


def test_hierarchical_swarm_multiple_loops():
    """Test HierarchicalSwarm with multiple feedback loops"""
    # Create agents for iterative refinement
    planner = Agent(
        agent_name="Strategic-Planner",
        agent_description="Strategic planning and project management",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    executor = Agent(
        agent_name="Task-Executor",
        agent_description="Task execution and implementation",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    reviewer = Agent(
        agent_name="Quality-Reviewer",
        agent_description="Quality assurance and review specialist",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    # Create swarm with multiple loops for iterative refinement
    swarm = HierarchicalSwarm(
        name="Iterative-Development-Swarm",
        description="Hierarchical swarm with iterative feedback loops",
        agents=[planner, executor, reviewer],
        max_loops=3,  # Allow multiple iterations
        verbose=True,
    )

    # Execute with multiple loops
    result = swarm.run(
        "Create a detailed project plan for implementing a machine learning recommendation system"
    )

    assert result is not None


def test_hierarchical_swarm_error_handling():
    """Test HierarchicalSwarm error handling"""
    # Test with empty agents list
    try:
        HierarchicalSwarm(agents=[])
        assert (
            False
        ), "Should have raised ValueError for empty agents list"
    except ValueError as e:
        assert "agents" in str(e).lower() or "empty" in str(e).lower()

    # Test with invalid max_loops
    researcher = Agent(
        agent_name="Test-Researcher",
        agent_description="Test researcher",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    try:
        HierarchicalSwarm(agents=[researcher], max_loops=0)
        assert (
            False
        ), "Should have raised ValueError for invalid max_loops"
    except ValueError as e:
        assert "max_loops" in str(e).lower() or "0" in str(e)


def test_hierarchical_swarm_collaboration_prompts():
    """Test HierarchicalSwarm with collaboration prompts enabled"""
    # Create agents
    data_analyst = Agent(
        agent_name="Data-Analyst",
        agent_description="Data analysis specialist",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    business_analyst = Agent(
        agent_name="Business-Analyst",
        agent_description="Business analysis specialist",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    # Create swarm with collaboration prompts
    swarm = HierarchicalSwarm(
        name="Collaborative-Analysis-Swarm",
        description="Hierarchical swarm with enhanced collaboration",
        agents=[data_analyst, business_analyst],
        max_loops=1,
        add_collaboration_prompt=True,
    )

    # Check that collaboration prompts were added to agents
    assert data_analyst.system_prompt is not None
    assert business_analyst.system_prompt is not None

    # Execute swarm
    result = swarm.run(
        "Analyze customer behavior patterns and provide business recommendations"
    )
    assert result is not None


def test_hierarchical_swarm_with_dashboard():
    """Test HierarchicalSwarm with interactive dashboard"""
    # Create agents
    content_creator = Agent(
        agent_name="Content-Creator",
        agent_description="Content creation specialist",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    editor = Agent(
        agent_name="Editor",
        agent_description="Content editor and proofreader",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    publisher = Agent(
        agent_name="Publisher",
        agent_description="Publishing and distribution specialist",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    # Create swarm with interactive dashboard
    swarm = HierarchicalSwarm(
        name="Content-Publishing-Swarm",
        description="Hierarchical swarm for content creation and publishing",
        agents=[content_creator, editor, publisher],
        max_loops=1,
        interactive=True,
        verbose=True,
    )

    # Verify dashboard was created
    assert swarm.dashboard is not None
    assert swarm.interactive is True

    # Execute swarm
    result = swarm.run(
        "Create a comprehensive guide on machine learning best practices"
    )
    assert result is not None


def test_hierarchical_swarm_real_world_scenario():
    """Test HierarchicalSwarm in a realistic business scenario"""
    # Create agents representing different business functions
    market_intelligence = Agent(
        agent_name="Market-Intelligence-Director",
        agent_description="Director of market intelligence and competitive analysis",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    product_strategy = Agent(
        agent_name="Product-Strategy-Manager",
        agent_description="Product strategy and roadmap manager",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    engineering_lead = Agent(
        agent_name="Engineering-Lead",
        agent_description="Senior engineering lead and technical architect",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    operations_manager = Agent(
        agent_name="Operations-Manager",
        agent_description="Operations and implementation manager",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    compliance_officer = Agent(
        agent_name="Compliance-Officer",
        agent_description="Legal compliance and regulatory specialist",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    # Create comprehensive hierarchical swarm
    swarm = HierarchicalSwarm(
        name="Enterprise-Strategy-Swarm",
        description="Enterprise-level strategic planning and execution swarm",
        agents=[
            market_intelligence,
            product_strategy,
            engineering_lead,
            operations_manager,
            compliance_officer,
        ],
        max_loops=2,
        verbose=True,
        add_collaboration_prompt=True,
    )

    # Test with complex enterprise scenario
    result = swarm.run(
        "Develop a comprehensive 5-year strategic plan for our company to become a leader in "
        "AI-powered enterprise solutions. Consider market opportunities, competitive landscape, "
        "technical requirements, operational capabilities, and regulatory compliance."
    )

    assert result is not None


def test_hierarchical_swarm_autosave_creates_workspace_dir(
    monkeypatch, tmp_path
):
    """Test that HierarchicalSwarm with autosave=True creates a workspace directory."""
    get_workspace_dir.cache_clear()
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path))

    agent1 = Agent(
        agent_name="Autosave-Hierarchical-1",
        agent_description="Agent for autosave test",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
    )
    agent2 = Agent(
        agent_name="Autosave-Hierarchical-2",
        agent_description="Agent for autosave test",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    swarm = HierarchicalSwarm(
        name="Autosave-Test-Swarm",
        description="Hierarchical swarm for autosave test",
        agents=[agent1, agent2],
        max_loops=1,
        autosave=True,
        verbose=False,
    )

    assert swarm.autosave is True
    assert swarm.swarm_workspace_dir is not None
    assert os.path.isdir(swarm.swarm_workspace_dir)
    assert "HierarchicalSwarm" in swarm.swarm_workspace_dir
    assert "Autosave-Test-Swarm" in swarm.swarm_workspace_dir

    get_workspace_dir.cache_clear()


def test_hierarchical_swarm_autosave_saves_conversation_after_run(
    monkeypatch, tmp_path
):
    """Test that HierarchicalSwarm saves conversation_history.json after run when autosave=True."""
    get_workspace_dir.cache_clear()
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path))

    agent1 = Agent(
        agent_name="Autosave-Run-Hierarchical-1",
        agent_description="Agent for autosave run test",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
    )
    agent2 = Agent(
        agent_name="Autosave-Run-Hierarchical-2",
        agent_description="Agent for autosave run test",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        print_on=False,
    )

    swarm = HierarchicalSwarm(
        name="Autosave-Run-Swarm",
        description="Hierarchical swarm for autosave run test",
        agents=[agent1, agent2],
        max_loops=1,
        autosave=True,
        verbose=False,
    )

    result = swarm.run(task="Say hello in one short sentence.")
    assert result is not None

    conversation_path = os.path.join(
        swarm.swarm_workspace_dir, "conversation_history.json"
    )
    assert os.path.isfile(
        conversation_path
    ), f"Expected conversation_history.json at {conversation_path}"

    get_workspace_dir.cache_clear()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
