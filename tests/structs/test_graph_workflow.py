import hashlib

import pytest

from swarms.structs.agent import Agent
from swarms.structs.graph_workflow import (
    GraphWorkflow,
    Node,
    NodeType,
)

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
        model_name="gpt-5.4",
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


def test_graph_workflow_checkpoint_writes_and_resumes(tmp_path):
    """Checkpoint files are written after each layer and skipped on second run."""
    from unittest.mock import MagicMock

    def make_agent(name):
        a = MagicMock()
        a.agent_name = name
        a.run = MagicMock(return_value=f"output-{name}")
        return a

    a1 = make_agent("CP-Alpha")
    a2 = make_agent("CP-Beta")
    a3 = make_agent("CP-Gamma")

    cp_dir = str(tmp_path / "checkpoints")
    wf = GraphWorkflow(name="CP-Test", checkpoint_dir=cp_dir)
    wf.add_nodes([a1, a2, a3])
    wf.add_edge("CP-Alpha", "CP-Beta")
    wf.add_edge("CP-Beta", "CP-Gamma")
    wf.compile()

    TASK = "checkpoint test task"

    # First run — all three agents execute, three checkpoint files written
    results = wf.run(TASK)
    assert results["CP-Alpha"] == "output-CP-Alpha"
    assert results["CP-Beta"] == "output-CP-Beta"
    assert results["CP-Gamma"] == "output-CP-Gamma"

    task_key = hashlib.sha256(TASK.encode("utf-8")).hexdigest()[:16]
    cp_files = list(tmp_path.glob("checkpoints/*.json"))
    assert len(cp_files) == 3
    assert any(f"{task_key}_layer_0" in f.name for f in cp_files)
    assert any(f"{task_key}_layer_1" in f.name for f in cp_files)
    assert any(f"{task_key}_layer_2" in f.name for f in cp_files)

    # Reset call counts, then run again — all layers should be skipped
    a1.run.reset_mock()
    a2.run.reset_mock()
    a3.run.reset_mock()

    results2 = wf.run(TASK)
    assert a1.run.call_count == 0
    assert a2.run.call_count == 0
    assert a3.run.call_count == 0
    assert results2["CP-Alpha"] == "output-CP-Alpha"


def test_graph_workflow_checkpoint_partial_resume(tmp_path):
    """When only some checkpoints exist, only the missing layers re-execute."""
    from unittest.mock import MagicMock

    def make_agent(name):
        a = MagicMock()
        a.agent_name = name
        a.run = MagicMock(return_value=f"output-{name}")
        return a

    a1 = make_agent("PR-Alpha")
    a2 = make_agent("PR-Beta")
    a3 = make_agent("PR-Gamma")

    cp_dir = str(tmp_path / "checkpoints")
    wf = GraphWorkflow(name="PR-Test", checkpoint_dir=cp_dir)
    wf.add_nodes([a1, a2, a3])
    wf.add_edge("PR-Alpha", "PR-Beta")
    wf.add_edge("PR-Beta", "PR-Gamma")
    wf.compile()

    TASK = "partial resume task"
    wf.run(TASK)

    # Delete the last layer's checkpoint to simulate a crash after layer 2
    task_key = hashlib.sha256(TASK.encode("utf-8")).hexdigest()[:16]
    (tmp_path / "checkpoints" / f"{task_key}_layer_2.json").unlink()

    a1.run.reset_mock()
    a2.run.reset_mock()
    a3.run.reset_mock()

    wf.run(TASK)

    assert a1.run.call_count == 0  # restored from checkpoint
    assert a2.run.call_count == 0  # restored from checkpoint
    assert a3.run.call_count == 1  # re-executed


def test_graph_workflow_clear_checkpoints(tmp_path):
    """clear_checkpoints() removes only the target task's files."""
    from unittest.mock import MagicMock

    def make_agent(name):
        a = MagicMock()
        a.agent_name = name
        a.run = MagicMock(return_value=f"output-{name}")
        return a

    a1 = make_agent("CL-Alpha")
    a2 = make_agent("CL-Beta")

    cp_dir = str(tmp_path / "checkpoints")
    wf = GraphWorkflow(name="CL-Test", checkpoint_dir=cp_dir)
    wf.add_nodes([a1, a2])
    wf.add_edge("CL-Alpha", "CL-Beta")
    wf.compile()

    TASK_A = "task a"
    TASK_B = "task b"

    wf.run(TASK_A)
    wf.run(TASK_B)

    all_files_before = list(tmp_path.glob("checkpoints/*.json"))
    assert len(all_files_before) == 4  # 2 layers x 2 tasks

    deleted = wf.clear_checkpoints(TASK_A)
    assert deleted == 2

    remaining = list(tmp_path.glob("checkpoints/*.json"))
    assert len(remaining) == 2  # only TASK_B files remain

    task_b_key = hashlib.sha256(TASK_B.encode("utf-8")).hexdigest()[
        :16
    ]
    assert all(task_b_key in f.name for f in remaining)


def test_graph_workflow_clear_checkpoints_no_dir():
    """clear_checkpoints() raises ValueError when checkpoint_dir is not set."""
    wf = GraphWorkflow(name="NoCp-Test")
    with pytest.raises(ValueError, match="checkpoint_dir"):
        wf.clear_checkpoints("some task")


def test_graph_workflow_checkpoint_conversation_replay(tmp_path):
    """Restored checkpoint outputs are replayed into self.conversation."""
    from unittest.mock import MagicMock

    def make_agent(name):
        a = MagicMock()
        a.agent_name = name
        a.run = MagicMock(return_value=f"output-{name}")
        return a

    a1 = make_agent("CV-Alpha")
    a2 = make_agent("CV-Beta")

    cp_dir = str(tmp_path / "checkpoints")
    wf = GraphWorkflow(name="CV-Test", checkpoint_dir=cp_dir)
    wf.add_nodes([a1, a2])
    wf.add_edge("CV-Alpha", "CV-Beta")
    wf.compile()

    TASK = "conversation replay task"
    wf.run(TASK)

    # Second run — both layers restored from checkpoints
    a1.run.reset_mock()
    a2.run.reset_mock()
    wf.conversation = type(wf.conversation)()  # fresh conversation
    wf.run(TASK)

    assert a1.run.call_count == 0
    assert a2.run.call_count == 0
    # Conversation should contain the restored outputs
    history_roles = [
        m["role"] for m in wf.conversation.conversation_history
    ]
    assert "CV-Alpha" in history_roles
    assert "CV-Beta" in history_roles


def test_graph_workflow_to_spec_round_trip():
    """to_spec / from_topology_spec round-trip preserves topology and metadata."""
    a = create_test_agent("Alpha", "First agent")
    b = create_test_agent("Beta", "Second agent")
    c = create_test_agent("Gamma", "Third agent")

    wf = GraphWorkflow(
        name="RoundTrip", description="Test pipeline", max_loops=2
    )
    wf.add_nodes([a, b, c])
    wf.add_edge("Alpha", "Beta", weight=1)
    wf.add_edge("Beta", "Gamma", weight=2)
    wf.compile()

    spec = wf.to_spec()

    # Top-level fields
    assert spec["name"] == "RoundTrip"
    assert spec["description"] == "Test pipeline"
    assert spec["max_loops"] == 2

    # Nodes are sorted by id
    node_ids = [n["id"] for n in spec["nodes"]]
    assert node_ids == sorted(node_ids)
    assert set(node_ids) == {"Alpha", "Beta", "Gamma"}
    for n in spec["nodes"]:
        assert n["agent_name"] == n["id"]

    # Edges are sorted by (source, target)
    edge_pairs = [(e["source"], e["target"]) for e in spec["edges"]]
    assert edge_pairs == sorted(edge_pairs)
    assert ("Alpha", "Beta") in edge_pairs
    assert ("Beta", "Gamma") in edge_pairs

    # entry / end points are sorted lists
    assert spec["entry_points"] == sorted(spec["entry_points"])
    assert spec["end_points"] == sorted(spec["end_points"])

    # Reconstruct
    registry = {"Alpha": a, "Beta": b, "Gamma": c}
    wf2 = GraphWorkflow.from_topology_spec(spec, registry)

    assert set(wf2.nodes.keys()) == {"Alpha", "Beta", "Gamma"}
    assert len(wf2.edges) == 2
    assert wf2.name == "RoundTrip"
    assert wf2.max_loops == 2
    assert set(wf2.entry_points) == set(wf.entry_points)
    assert set(wf2.end_points) == set(wf.end_points)

    # Agent objects are resolved correctly
    assert wf2.nodes["Alpha"].agent is a
    assert wf2.nodes["Beta"].agent is b
    assert wf2.nodes["Gamma"].agent is c


def test_graph_workflow_to_spec_deterministic_order():
    """to_spec output is identical regardless of insertion order."""
    a = create_test_agent("Zebra")
    b = create_test_agent("Apple")
    c = create_test_agent("Mango")

    wf1 = GraphWorkflow(name="Order-Test")
    wf1.add_nodes([a, b, c])
    wf1.add_edge("Apple", "Mango")
    wf1.add_edge("Mango", "Zebra")
    wf1.compile()

    wf2 = GraphWorkflow(name="Order-Test")
    wf2.add_nodes([c, a, b])  # different insertion order
    wf2.add_edge("Mango", "Zebra")
    wf2.add_edge("Apple", "Mango")
    wf2.compile()

    assert wf1.to_spec()["nodes"] == wf2.to_spec()["nodes"]
    assert wf1.to_spec()["edges"] == wf2.to_spec()["edges"]


def test_graph_workflow_to_spec_node_metadata():
    """Node metadata is preserved through the spec round-trip."""
    a = create_test_agent("Alpha")
    b = create_test_agent("Beta")

    from swarms.structs.graph_workflow import Node

    wf = GraphWorkflow(name="Meta-Test")
    wf.nodes["Alpha"] = Node(
        id="Alpha", agent=a, metadata={"role": "lead", "priority": 1}
    )
    wf.nodes["Beta"] = Node(
        id="Beta", agent=b, metadata={"role": "support"}
    )
    wf.add_edge("Alpha", "Beta")
    wf.compile()

    spec = wf.to_spec()
    alpha_spec = next(n for n in spec["nodes"] if n["id"] == "Alpha")
    assert alpha_spec["metadata"] == {"role": "lead", "priority": 1}

    registry = {"Alpha": a, "Beta": b}
    wf2 = GraphWorkflow.from_topology_spec(spec, registry)
    assert wf2.nodes["Alpha"].metadata == {
        "role": "lead",
        "priority": 1,
    }
    assert wf2.nodes["Beta"].metadata == {"role": "support"}


def test_graph_workflow_from_topology_spec_missing_agent():
    """from_topology_spec raises ValueError when an agent is absent from the registry."""
    a = create_test_agent("Alpha")
    b = create_test_agent("Beta")

    wf = GraphWorkflow(name="Missing-Agent-Test")
    wf.add_nodes([a, b])
    wf.add_edge("Alpha", "Beta")
    wf.compile()

    spec = wf.to_spec()

    # Registry is missing "Beta"
    with pytest.raises(ValueError, match="Beta"):
        GraphWorkflow.from_topology_spec(spec, {"Alpha": a})


def test_graph_workflow_from_topology_spec_malformed_node():
    """from_topology_spec raises ValueError when a node dict is missing required keys."""
    spec = {
        "nodes": [{"id": "Alpha"}],  # missing "agent_name"
        "edges": [],
    }
    with pytest.raises(ValueError, match="agent_name"):
        GraphWorkflow.from_topology_spec(spec, {})


def test_graph_workflow_from_topology_spec_malformed_edge():
    """from_topology_spec raises ValueError when an edge dict is missing required keys."""
    a = create_test_agent("Alpha")
    spec = {
        "nodes": [{"id": "Alpha", "agent_name": "Alpha"}],
        "edges": [{"source": "Alpha"}],  # missing "target"
    }
    with pytest.raises(ValueError, match="target"):
        GraphWorkflow.from_topology_spec(spec, {"Alpha": a})


def test_graph_workflow_from_topology_spec_not_a_dict():
    """from_topology_spec raises ValueError when spec is not a dict."""
    with pytest.raises(ValueError, match="dict"):
        GraphWorkflow.from_topology_spec("not-a-dict", {})


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


@pytest.mark.parametrize("backend", ["networkx", "rustworkx"])
def test_graph_workflow_max_loops_accumulates_results(backend):
    """Test that max_loops > 1 actually executes multiple iterations and
    accumulates results across loops (fixes #1481)."""
    if backend == "rustworkx" and not RUSTWORKX_AVAILABLE:
        pytest.skip("rustworkx not available")

    agent1 = create_test_agent("Agent1", "Entry agent")
    agent2 = create_test_agent("Agent2", "End agent")

    workflow = GraphWorkflow(
        name=f"MultiLoop-Test-{backend}",
        backend=backend,
        max_loops=3,
    )
    workflow.add_node(agent1)
    workflow.add_node(agent2)
    workflow.add_edge(agent1, agent2)

    result = workflow.run("Iteratively refine analysis")
    assert result is not None

    # With max_loops > 1, result should contain per-loop keys
    assert "Agent1_loop_1" in result
    assert "Agent2_loop_1" in result
    assert "Agent1_loop_2" in result
    assert "Agent2_loop_2" in result
    assert "Agent1_loop_3" in result
    assert "Agent2_loop_3" in result

    # Final loop results should also be accessible under plain node IDs
    assert "Agent1" in result
    assert "Agent2" in result


def test_graph_workflow_single_loop_backward_compatible():
    """Test that max_loops=1 (the default) returns results in the original
    format — plain node-ID keys, no loop suffixes."""
    agent1 = create_test_agent("Agent1", "Entry agent")
    agent2 = create_test_agent("Agent2", "End agent")

    workflow = GraphWorkflow(name="SingleLoop-Compat")
    workflow.add_node(agent1)
    workflow.add_node(agent2)
    workflow.add_edge(agent1, agent2)

    result = workflow.run("Simple task")
    assert result is not None
    assert "Agent1" in result
    assert "Agent2" in result

    # Should NOT have loop-suffixed keys
    assert not any(
        k.endswith("_loop_1") for k in result
    ), "Single-loop results should not contain loop-suffixed keys"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
