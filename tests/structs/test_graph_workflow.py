import pytest
from swarms.structs.graph_workflow import (
    GraphWorkflow,
    Node,
    NodeType,
)
from swarms.structs.agent import Agent

try:
    import rustworkx as rx

    RUSTWORKX_AVAILABLE = True
except ImportError:
    RUSTWORKX_AVAILABLE = False


def create_test_agent(name: str, description: str = None) -> Agent:
    """Create a real agent for testing"""
    if description is None:
        description = f"Test agent for {name} operations"

    return Agent(
        agent_name=name,
        agent_description=description,
        model_name="gpt-4o-mini",
        verbose=False,
        print_on=False,
        max_loops=1,
    )


def test_graph_workflow_basic_node_creation():
    """Test basic GraphWorkflow node creation with real agents"""
    # Test basic node creation
    agent = create_test_agent(
        "TestAgent", "Test agent for node creation"
    )
    node = Node.from_agent(agent)
    assert node.id == "TestAgent"
    assert node.type == NodeType.AGENT
    assert node.agent == agent

    # Test node with custom id
    node2 = Node(id="CustomID", type=NodeType.AGENT, agent=agent)
    assert node2.id == "CustomID"


def test_graph_workflow_multi_agent_collaboration():
    """Test GraphWorkflow with multiple agents in a collaboration scenario"""
    # Create specialized agents for a business analysis workflow
    market_researcher = create_test_agent(
        "Market-Researcher",
        "Specialist in market analysis and trend identification",
    )

    data_analyst = create_test_agent(
        "Data-Analyst",
        "Expert in data processing and statistical analysis",
    )

    strategy_consultant = create_test_agent(
        "Strategy-Consultant",
        "Senior consultant for strategic planning and recommendations",
    )

    # Create workflow with linear execution path
    workflow = GraphWorkflow(name="Business-Analysis-Workflow")
    workflow.add_node(market_researcher)
    workflow.add_node(data_analyst)
    workflow.add_node(strategy_consultant)

    # Add edges to define execution order
    workflow.add_edge("Market-Researcher", "Data-Analyst")
    workflow.add_edge("Data-Analyst", "Strategy-Consultant")

    # Test workflow execution
    result = workflow.run(
        "Analyze market opportunities for AI in healthcare"
    )
    assert result is not None


def test_graph_workflow_parallel_execution():
    """Test GraphWorkflow with parallel execution paths"""
    # Create agents for parallel analysis
    technical_analyst = create_test_agent(
        "Technical-Analyst",
        "Technical feasibility and implementation analysis",
    )

    market_analyst = create_test_agent(
        "Market-Analyst",
        "Market positioning and competitive analysis",
    )

    financial_analyst = create_test_agent(
        "Financial-Analyst", "Financial modeling and ROI analysis"
    )

    risk_assessor = create_test_agent(
        "Risk-Assessor", "Risk assessment and mitigation planning"
    )

    # Create workflow with parallel execution
    workflow = GraphWorkflow(name="Parallel-Analysis-Workflow")
    workflow.add_node(technical_analyst)
    workflow.add_node(market_analyst)
    workflow.add_node(financial_analyst)
    workflow.add_node(risk_assessor)

    # Add edges for fan-out execution (one to many)
    workflow.add_edges_from_source(
        "Technical-Analyst",
        ["Market-Analyst", "Financial-Analyst", "Risk-Assessor"],
    )

    # Test parallel execution
    result = workflow.run(
        "Evaluate feasibility of launching a new fintech platform"
    )
    assert result is not None


def test_graph_workflow_complex_topology():
    """Test GraphWorkflow with complex node topology"""
    # Create agents for a comprehensive product development workflow
    product_manager = create_test_agent(
        "Product-Manager", "Product strategy and roadmap management"
    )

    ux_designer = create_test_agent(
        "UX-Designer", "User experience design and research"
    )

    backend_developer = create_test_agent(
        "Backend-Developer",
        "Backend system architecture and development",
    )

    frontend_developer = create_test_agent(
        "Frontend-Developer",
        "Frontend interface and user interaction development",
    )

    qa_engineer = create_test_agent(
        "QA-Engineer", "Quality assurance and testing specialist"
    )

    devops_engineer = create_test_agent(
        "DevOps-Engineer", "Deployment and infrastructure management"
    )

    # Create workflow with complex dependencies
    workflow = GraphWorkflow(name="Product-Development-Workflow")
    workflow.add_node(product_manager)
    workflow.add_node(ux_designer)
    workflow.add_node(backend_developer)
    workflow.add_node(frontend_developer)
    workflow.add_node(qa_engineer)
    workflow.add_node(devops_engineer)

    # Define complex execution topology
    workflow.add_edge("Product-Manager", "UX-Designer")
    workflow.add_edge("UX-Designer", "Frontend-Developer")
    workflow.add_edge("Product-Manager", "Backend-Developer")
    workflow.add_edge("Backend-Developer", "QA-Engineer")
    workflow.add_edge("Frontend-Developer", "QA-Engineer")
    workflow.add_edge("QA-Engineer", "DevOps-Engineer")

    # Test complex workflow execution
    result = workflow.run(
        "Develop a comprehensive e-commerce platform with AI recommendations"
    )
    assert result is not None


def test_graph_workflow_error_handling():
    """Test GraphWorkflow error handling and validation"""
    # Test with empty workflow
    workflow = GraphWorkflow()
    result = workflow.run("Test task")
    # Empty workflow should handle gracefully
    assert result is not None

    # Test workflow compilation and caching
    researcher = create_test_agent(
        "Researcher", "Research specialist"
    )
    workflow.add_node(researcher)

    # First run should compile
    result1 = workflow.run("Research task")
    assert result1 is not None

    # Second run should use cached compilation
    result2 = workflow.run("Another research task")
    assert result2 is not None


def test_graph_workflow_node_metadata():
    """Test GraphWorkflow with node metadata"""
    # Create agents with different priorities and requirements
    high_priority_agent = create_test_agent(
        "High-Priority-Analyst", "High priority analysis specialist"
    )

    standard_agent = create_test_agent(
        "Standard-Analyst", "Standard analysis agent"
    )

    # Create workflow and add nodes with metadata
    workflow = GraphWorkflow(name="Metadata-Workflow")
    workflow.add_node(
        high_priority_agent,
        metadata={"priority": "high", "timeout": 60},
    )
    workflow.add_node(
        standard_agent, metadata={"priority": "normal", "timeout": 30}
    )

    # Add execution dependency
    workflow.add_edge("High-Priority-Analyst", "Standard-Analyst")

    # Test execution with metadata
    result = workflow.run(
        "Analyze business requirements with different priorities"
    )
    assert result is not None


@pytest.mark.parametrize("backend", ["networkx", "rustworkx"])
def test_graph_workflow_backend_basic(backend):
    """Test GraphWorkflow basic functionality with both backends"""
    if backend == "rustworkx" and not RUSTWORKX_AVAILABLE:
        pytest.skip("rustworkx not available")

    agent1 = create_test_agent("Agent1", "First agent")
    agent2 = create_test_agent("Agent2", "Second agent")

    workflow = GraphWorkflow(
        name=f"Backend-Test-{backend}", backend=backend
    )
    workflow.add_node(agent1)
    workflow.add_node(agent2)
    workflow.add_edge(agent1, agent2)

    assert len(workflow.nodes) == 2
    assert len(workflow.edges) == 1

    result = workflow.run("Test task")
    assert result is not None
    assert "Agent1" in result
    assert "Agent2" in result


@pytest.mark.parametrize("backend", ["networkx", "rustworkx"])
def test_graph_workflow_backend_parallel_execution(backend):
    """Test parallel execution with both backends"""
    if backend == "rustworkx" and not RUSTWORKX_AVAILABLE:
        pytest.skip("rustworkx not available")

    coordinator = create_test_agent(
        "Coordinator", "Coordinates tasks"
    )
    analyst1 = create_test_agent("Analyst1", "First analyst")
    analyst2 = create_test_agent("Analyst2", "Second analyst")
    analyst3 = create_test_agent("Analyst3", "Third analyst")

    workflow = GraphWorkflow(
        name=f"Parallel-Test-{backend}", backend=backend
    )
    workflow.add_node(coordinator)
    workflow.add_node(analyst1)
    workflow.add_node(analyst2)
    workflow.add_node(analyst3)

    workflow.add_edges_from_source(
        coordinator, [analyst1, analyst2, analyst3]
    )

    workflow.compile()
    assert len(workflow._sorted_layers) >= 1
    assert (
        len(workflow._sorted_layers[0]) == 1
    )  # Coordinator in first layer

    result = workflow.run("Analyze data in parallel")
    assert result is not None


@pytest.mark.parametrize("backend", ["networkx", "rustworkx"])
def test_graph_workflow_backend_fan_in_pattern(backend):
    """Test fan-in pattern with both backends"""
    if backend == "rustworkx" and not RUSTWORKX_AVAILABLE:
        pytest.skip("rustworkx not available")

    analyst1 = create_test_agent("Analyst1", "First analyst")
    analyst2 = create_test_agent("Analyst2", "Second analyst")
    analyst3 = create_test_agent("Analyst3", "Third analyst")
    synthesizer = create_test_agent(
        "Synthesizer", "Synthesizes results"
    )

    workflow = GraphWorkflow(
        name=f"FanIn-Test-{backend}", backend=backend
    )
    workflow.add_node(analyst1)
    workflow.add_node(analyst2)
    workflow.add_node(analyst3)
    workflow.add_node(synthesizer)

    workflow.add_edges_to_target(
        [analyst1, analyst2, analyst3], synthesizer
    )

    workflow.compile()
    assert len(workflow._sorted_layers) >= 2
    assert synthesizer.agent_name in workflow.end_points

    result = workflow.run("Synthesize multiple analyses")
    assert result is not None


@pytest.mark.parametrize("backend", ["networkx", "rustworkx"])
def test_graph_workflow_backend_parallel_chain(backend):
    """Test parallel chain pattern with both backends"""
    if backend == "rustworkx" and not RUSTWORKX_AVAILABLE:
        pytest.skip("rustworkx not available")

    collector1 = create_test_agent("Collector1", "First collector")
    collector2 = create_test_agent("Collector2", "Second collector")
    processor1 = create_test_agent("Processor1", "First processor")
    processor2 = create_test_agent("Processor2", "Second processor")

    workflow = GraphWorkflow(
        name=f"ParallelChain-Test-{backend}", backend=backend
    )
    workflow.add_node(collector1)
    workflow.add_node(collector2)
    workflow.add_node(processor1)
    workflow.add_node(processor2)

    workflow.add_parallel_chain(
        [collector1, collector2], [processor1, processor2]
    )

    workflow.compile()
    assert len(workflow.edges) == 4  # 2x2 = 4 edges

    result = workflow.run("Process data from multiple collectors")
    assert result is not None


@pytest.mark.parametrize("backend", ["networkx", "rustworkx"])
def test_graph_workflow_backend_complex_topology(backend):
    """Test complex topology with both backends"""
    if backend == "rustworkx" and not RUSTWORKX_AVAILABLE:
        pytest.skip("rustworkx not available")

    agents = [
        create_test_agent(f"Agent{i}", f"Agent {i}") for i in range(5)
    ]

    workflow = GraphWorkflow(
        name=f"Complex-Topology-{backend}", backend=backend
    )
    for agent in agents:
        workflow.add_node(agent)

    workflow.add_edge(agents[0], agents[1])
    workflow.add_edge(agents[0], agents[2])
    workflow.add_edge(agents[1], agents[3])
    workflow.add_edge(agents[2], agents[3])
    workflow.add_edge(agents[3], agents[4])

    workflow.compile()
    assert len(workflow._sorted_layers) >= 3

    result = workflow.run("Execute complex workflow")
    assert result is not None


@pytest.mark.parametrize("backend", ["networkx", "rustworkx"])
def test_graph_workflow_backend_validation(backend):
    """Test workflow validation with both backends"""
    if backend == "rustworkx" and not RUSTWORKX_AVAILABLE:
        pytest.skip("rustworkx not available")

    agent1 = create_test_agent("Agent1", "First agent")
    agent2 = create_test_agent("Agent2", "Second agent")
    isolated = create_test_agent("Isolated", "Isolated agent")

    workflow = GraphWorkflow(
        name=f"Validation-Test-{backend}", backend=backend
    )
    workflow.add_node(agent1)
    workflow.add_node(agent2)
    workflow.add_node(isolated)
    workflow.add_edge(agent1, agent2)

    validation = workflow.validate(auto_fix=False)
    assert isinstance(validation, dict)
    assert "is_valid" in validation

    validation_fixed = workflow.validate(auto_fix=True)
    assert isinstance(validation_fixed, dict)


@pytest.mark.parametrize("backend", ["networkx", "rustworkx"])
def test_graph_workflow_backend_entry_end_points(backend):
    """Test entry and end points with both backends"""
    if backend == "rustworkx" and not RUSTWORKX_AVAILABLE:
        pytest.skip("rustworkx not available")

    agent1 = create_test_agent("Agent1", "Entry agent")
    agent2 = create_test_agent("Agent2", "Middle agent")
    agent3 = create_test_agent("Agent3", "End agent")

    workflow = GraphWorkflow(
        name=f"EntryEnd-Test-{backend}", backend=backend
    )
    workflow.add_node(agent1)
    workflow.add_node(agent2)
    workflow.add_node(agent3)
    workflow.add_edge(agent1, agent2)
    workflow.add_edge(agent2, agent3)

    workflow.auto_set_entry_points()
    workflow.auto_set_end_points()

    assert agent1.agent_name in workflow.entry_points
    assert agent3.agent_name in workflow.end_points


def test_graph_workflow_rustworkx_specific():
    """Test rustworkx-specific features"""
    if not RUSTWORKX_AVAILABLE:
        pytest.skip("rustworkx not available")

    agent1 = create_test_agent("Agent1", "First agent")
    agent2 = create_test_agent("Agent2", "Second agent")
    agent3 = create_test_agent("Agent3", "Third agent")

    workflow = GraphWorkflow(
        name="Rustworkx-Specific-Test", backend="rustworkx"
    )
    workflow.add_node(agent1)
    workflow.add_node(agent2)
    workflow.add_node(agent3)
    workflow.add_edge(agent1, agent2)
    workflow.add_edge(agent2, agent3)

    assert (
        workflow.graph_backend.__class__.__name__
        == "RustworkxBackend"
    )
    assert hasattr(workflow.graph_backend, "_node_id_to_index")
    assert hasattr(workflow.graph_backend, "_index_to_node_id")

    workflow.compile()
    assert len(workflow._sorted_layers) == 3

    predecessors = list(
        workflow.graph_backend.predecessors(agent2.agent_name)
    )
    assert agent1.agent_name in predecessors

    descendants = workflow.graph_backend.descendants(
        agent1.agent_name
    )
    assert agent2.agent_name in descendants
    assert agent3.agent_name in descendants

    result = workflow.run("Test rustworkx backend")
    assert result is not None


def test_graph_workflow_rustworkx_large_scale():
    """Test rustworkx with larger workflow"""
    if not RUSTWORKX_AVAILABLE:
        pytest.skip("rustworkx not available")

    agents = [
        create_test_agent(f"Agent{i}", f"Agent {i}")
        for i in range(10)
    ]

    workflow = GraphWorkflow(
        name="Rustworkx-Large-Scale", backend="rustworkx"
    )
    for agent in agents:
        workflow.add_node(agent)

    for i in range(len(agents) - 1):
        workflow.add_edge(agents[i], agents[i + 1])

    workflow.compile()
    assert len(workflow._sorted_layers) == 10

    result = workflow.run("Test large scale workflow")
    assert result is not None
    assert len(result) == 10


def test_graph_workflow_rustworkx_agent_objects():
    """Test rustworkx with Agent objects directly in edges"""
    if not RUSTWORKX_AVAILABLE:
        pytest.skip("rustworkx not available")

    agent1 = create_test_agent("Agent1", "First agent")
    agent2 = create_test_agent("Agent2", "Second agent")
    agent3 = create_test_agent("Agent3", "Third agent")

    workflow = GraphWorkflow(
        name="Rustworkx-Agent-Objects", backend="rustworkx"
    )
    workflow.add_node(agent1)
    workflow.add_node(agent2)
    workflow.add_node(agent3)

    workflow.add_edges_from_source(agent1, [agent2, agent3])
    workflow.add_edges_to_target([agent2, agent3], agent1)

    workflow.compile()
    assert len(workflow.edges) == 4

    result = workflow.run("Test agent objects in edges")
    assert result is not None


def test_graph_workflow_backend_fallback():
    """Test backend fallback when rustworkx unavailable"""
    workflow = GraphWorkflow(
        name="Fallback-Test", backend="rustworkx"
    )
    agent = create_test_agent("Agent", "Test agent")
    workflow.add_node(agent)

    if not RUSTWORKX_AVAILABLE:
        assert (
            workflow.graph_backend.__class__.__name__
            == "NetworkXBackend"
        )
    else:
        assert (
            workflow.graph_backend.__class__.__name__
            == "RustworkxBackend"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
