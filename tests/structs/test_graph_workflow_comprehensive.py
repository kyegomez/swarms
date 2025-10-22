"""
Comprehensive Testing Suite for GraphWorkflow

This module provides thorough testing of all GraphWorkflow functionality including:
- Node and Edge creation and manipulation
- Workflow construction and compilation
- Execution with various parameters
- Multi-agent collaboration scenarios
- Error handling and edge cases

Tests follow the example.py pattern with real agents and multiple agent scenarios.
"""

from swarms.structs.graph_workflow import (
    GraphWorkflow,
    Node,
    NodeType,
)
from swarms.structs.agent import Agent


def create_test_agent(name: str, description: str = None) -> Agent:
    """Create a real agent for testing"""
    if description is None:
        description = f"Test agent for {name} operations"

    return Agent(
        agent_name=name,
        agent_description=description,
        model_name="gpt-4o",
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
