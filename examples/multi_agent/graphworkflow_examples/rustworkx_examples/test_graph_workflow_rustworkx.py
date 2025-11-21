import pytest
from swarms.structs.graph_workflow import (
    GraphWorkflow,
)
from swarms.structs.agent import Agent

try:
    import rustworkx as rx

    RUSTWORKX_AVAILABLE = True
except ImportError:
    RUSTWORKX_AVAILABLE = False


def create_test_agent(name: str, description: str = None) -> Agent:
    """Create a test agent"""
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


@pytest.mark.skipif(
    not RUSTWORKX_AVAILABLE, reason="rustworkx not available"
)
class TestRustworkxBackend:
    """Test suite for rustworkx backend"""

    def test_rustworkx_backend_initialization(self):
        """Test that rustworkx backend is properly initialized"""
        workflow = GraphWorkflow(name="Test", backend="rustworkx")
        assert (
            workflow.graph_backend.__class__.__name__
            == "RustworkxBackend"
        )
        assert hasattr(workflow.graph_backend, "_node_id_to_index")
        assert hasattr(workflow.graph_backend, "_index_to_node_id")
        assert hasattr(workflow.graph_backend, "graph")

    def test_rustworkx_node_addition(self):
        """Test adding nodes to rustworkx backend"""
        workflow = GraphWorkflow(name="Test", backend="rustworkx")
        agent = create_test_agent("TestAgent", "Test agent")

        workflow.add_node(agent)

        assert "TestAgent" in workflow.nodes
        assert "TestAgent" in workflow.graph_backend._node_id_to_index
        assert (
            workflow.graph_backend._node_id_to_index["TestAgent"]
            in workflow.graph_backend._index_to_node_id
        )

    def test_rustworkx_edge_addition(self):
        """Test adding edges to rustworkx backend"""
        workflow = GraphWorkflow(name="Test", backend="rustworkx")
        agent1 = create_test_agent("Agent1", "First agent")
        agent2 = create_test_agent("Agent2", "Second agent")

        workflow.add_node(agent1)
        workflow.add_node(agent2)
        workflow.add_edge(agent1, agent2)

        assert len(workflow.edges) == 1
        assert workflow.edges[0].source == "Agent1"
        assert workflow.edges[0].target == "Agent2"

    def test_rustworkx_topological_generations_linear(self):
        """Test topological generations with linear chain"""
        workflow = GraphWorkflow(
            name="Linear-Test", backend="rustworkx"
        )
        agents = [
            create_test_agent(f"Agent{i}", f"Agent {i}")
            for i in range(5)
        ]

        for agent in agents:
            workflow.add_node(agent)

        for i in range(len(agents) - 1):
            workflow.add_edge(agents[i], agents[i + 1])

        workflow.compile()

        assert len(workflow._sorted_layers) == 5
        assert workflow._sorted_layers[0] == ["Agent0"]
        assert workflow._sorted_layers[1] == ["Agent1"]
        assert workflow._sorted_layers[2] == ["Agent2"]
        assert workflow._sorted_layers[3] == ["Agent3"]
        assert workflow._sorted_layers[4] == ["Agent4"]

    def test_rustworkx_topological_generations_fan_out(self):
        """Test topological generations with fan-out pattern"""
        workflow = GraphWorkflow(
            name="FanOut-Test", backend="rustworkx"
        )
        coordinator = create_test_agent("Coordinator", "Coordinates")
        analyst1 = create_test_agent("Analyst1", "First analyst")
        analyst2 = create_test_agent("Analyst2", "Second analyst")
        analyst3 = create_test_agent("Analyst3", "Third analyst")

        workflow.add_node(coordinator)
        workflow.add_node(analyst1)
        workflow.add_node(analyst2)
        workflow.add_node(analyst3)

        workflow.add_edges_from_source(
            coordinator, [analyst1, analyst2, analyst3]
        )

        workflow.compile()

        assert len(workflow._sorted_layers) == 2
        assert len(workflow._sorted_layers[0]) == 1
        assert "Coordinator" in workflow._sorted_layers[0]
        assert len(workflow._sorted_layers[1]) == 3
        assert "Analyst1" in workflow._sorted_layers[1]
        assert "Analyst2" in workflow._sorted_layers[1]
        assert "Analyst3" in workflow._sorted_layers[1]

    def test_rustworkx_topological_generations_fan_in(self):
        """Test topological generations with fan-in pattern"""
        workflow = GraphWorkflow(
            name="FanIn-Test", backend="rustworkx"
        )
        analyst1 = create_test_agent("Analyst1", "First analyst")
        analyst2 = create_test_agent("Analyst2", "Second analyst")
        analyst3 = create_test_agent("Analyst3", "Third analyst")
        synthesizer = create_test_agent("Synthesizer", "Synthesizes")

        workflow.add_node(analyst1)
        workflow.add_node(analyst2)
        workflow.add_node(analyst3)
        workflow.add_node(synthesizer)

        workflow.add_edges_to_target(
            [analyst1, analyst2, analyst3], synthesizer
        )

        workflow.compile()

        assert len(workflow._sorted_layers) == 2
        assert len(workflow._sorted_layers[0]) == 3
        assert "Analyst1" in workflow._sorted_layers[0]
        assert "Analyst2" in workflow._sorted_layers[0]
        assert "Analyst3" in workflow._sorted_layers[0]
        assert len(workflow._sorted_layers[1]) == 1
        assert "Synthesizer" in workflow._sorted_layers[1]

    def test_rustworkx_topological_generations_complex(self):
        """Test topological generations with complex topology"""
        workflow = GraphWorkflow(
            name="Complex-Test", backend="rustworkx"
        )
        agents = [
            create_test_agent(f"Agent{i}", f"Agent {i}")
            for i in range(6)
        ]

        for agent in agents:
            workflow.add_node(agent)

        # Create: Agent0 -> Agent1, Agent2
        #         Agent1, Agent2 -> Agent3
        #         Agent3 -> Agent4, Agent5
        workflow.add_edge(agents[0], agents[1])
        workflow.add_edge(agents[0], agents[2])
        workflow.add_edge(agents[1], agents[3])
        workflow.add_edge(agents[2], agents[3])
        workflow.add_edge(agents[3], agents[4])
        workflow.add_edge(agents[3], agents[5])

        workflow.compile()

        assert len(workflow._sorted_layers) == 4
        assert "Agent0" in workflow._sorted_layers[0]
        assert (
            "Agent1" in workflow._sorted_layers[1]
            or "Agent2" in workflow._sorted_layers[1]
        )
        assert "Agent3" in workflow._sorted_layers[2]
        assert (
            "Agent4" in workflow._sorted_layers[3]
            or "Agent5" in workflow._sorted_layers[3]
        )

    def test_rustworkx_predecessors(self):
        """Test predecessor retrieval"""
        workflow = GraphWorkflow(
            name="Predecessors-Test", backend="rustworkx"
        )
        agent1 = create_test_agent("Agent1", "First agent")
        agent2 = create_test_agent("Agent2", "Second agent")
        agent3 = create_test_agent("Agent3", "Third agent")

        workflow.add_node(agent1)
        workflow.add_node(agent2)
        workflow.add_node(agent3)

        workflow.add_edge(agent1, agent2)
        workflow.add_edge(agent2, agent3)

        predecessors = list(
            workflow.graph_backend.predecessors("Agent2")
        )
        assert "Agent1" in predecessors
        assert len(predecessors) == 1

        predecessors = list(
            workflow.graph_backend.predecessors("Agent3")
        )
        assert "Agent2" in predecessors
        assert len(predecessors) == 1

        predecessors = list(
            workflow.graph_backend.predecessors("Agent1")
        )
        assert len(predecessors) == 0

    def test_rustworkx_descendants(self):
        """Test descendant retrieval"""
        workflow = GraphWorkflow(
            name="Descendants-Test", backend="rustworkx"
        )
        agent1 = create_test_agent("Agent1", "First agent")
        agent2 = create_test_agent("Agent2", "Second agent")
        agent3 = create_test_agent("Agent3", "Third agent")

        workflow.add_node(agent1)
        workflow.add_node(agent2)
        workflow.add_node(agent3)

        workflow.add_edge(agent1, agent2)
        workflow.add_edge(agent2, agent3)

        descendants = workflow.graph_backend.descendants("Agent1")
        assert "Agent2" in descendants
        assert "Agent3" in descendants
        assert len(descendants) == 2

        descendants = workflow.graph_backend.descendants("Agent2")
        assert "Agent3" in descendants
        assert len(descendants) == 1

        descendants = workflow.graph_backend.descendants("Agent3")
        assert len(descendants) == 0

    def test_rustworkx_in_degree(self):
        """Test in-degree calculation"""
        workflow = GraphWorkflow(
            name="InDegree-Test", backend="rustworkx"
        )
        agent1 = create_test_agent("Agent1", "First agent")
        agent2 = create_test_agent("Agent2", "Second agent")
        agent3 = create_test_agent("Agent3", "Third agent")

        workflow.add_node(agent1)
        workflow.add_node(agent2)
        workflow.add_node(agent3)

        workflow.add_edge(agent1, agent2)
        workflow.add_edge(agent3, agent2)

        assert workflow.graph_backend.in_degree("Agent1") == 0
        assert workflow.graph_backend.in_degree("Agent2") == 2
        assert workflow.graph_backend.in_degree("Agent3") == 0

    def test_rustworkx_out_degree(self):
        """Test out-degree calculation"""
        workflow = GraphWorkflow(
            name="OutDegree-Test", backend="rustworkx"
        )
        agent1 = create_test_agent("Agent1", "First agent")
        agent2 = create_test_agent("Agent2", "Second agent")
        agent3 = create_test_agent("Agent3", "Third agent")

        workflow.add_node(agent1)
        workflow.add_node(agent2)
        workflow.add_node(agent3)

        workflow.add_edge(agent1, agent2)
        workflow.add_edge(agent1, agent3)

        assert workflow.graph_backend.out_degree("Agent1") == 2
        assert workflow.graph_backend.out_degree("Agent2") == 0
        assert workflow.graph_backend.out_degree("Agent3") == 0

    def test_rustworkx_agent_objects_in_edges(self):
        """Test using Agent objects directly in edge methods"""
        workflow = GraphWorkflow(
            name="AgentObjects-Test", backend="rustworkx"
        )
        agent1 = create_test_agent("Agent1", "First agent")
        agent2 = create_test_agent("Agent2", "Second agent")
        agent3 = create_test_agent("Agent3", "Third agent")

        workflow.add_node(agent1)
        workflow.add_node(agent2)
        workflow.add_node(agent3)

        # Use Agent objects directly
        workflow.add_edges_from_source(agent1, [agent2, agent3])
        workflow.add_edges_to_target([agent2, agent3], agent1)

        workflow.compile()

        assert len(workflow.edges) == 4
        assert len(workflow._sorted_layers) >= 1

    def test_rustworkx_parallel_chain(self):
        """Test parallel chain pattern"""
        workflow = GraphWorkflow(
            name="ParallelChain-Test", backend="rustworkx"
        )
        sources = [
            create_test_agent(f"Source{i}", f"Source {i}")
            for i in range(3)
        ]
        targets = [
            create_test_agent(f"Target{i}", f"Target {i}")
            for i in range(3)
        ]

        for agent in sources + targets:
            workflow.add_node(agent)

        workflow.add_parallel_chain(sources, targets)

        workflow.compile()

        assert len(workflow.edges) == 9  # 3x3 = 9 edges
        assert len(workflow._sorted_layers) == 2

    def test_rustworkx_large_scale(self):
        """Test rustworkx with large workflow"""
        workflow = GraphWorkflow(
            name="LargeScale-Test", backend="rustworkx"
        )
        agents = [
            create_test_agent(f"Agent{i}", f"Agent {i}")
            for i in range(20)
        ]

        for agent in agents:
            workflow.add_node(agent)

        # Create linear chain
        for i in range(len(agents) - 1):
            workflow.add_edge(agents[i], agents[i + 1])

        workflow.compile()

        assert len(workflow._sorted_layers) == 20
        assert len(workflow.nodes) == 20
        assert len(workflow.edges) == 19

    def test_rustworkx_reverse(self):
        """Test graph reversal"""
        workflow = GraphWorkflow(
            name="Reverse-Test", backend="rustworkx"
        )
        agent1 = create_test_agent("Agent1", "First agent")
        agent2 = create_test_agent("Agent2", "Second agent")

        workflow.add_node(agent1)
        workflow.add_node(agent2)
        workflow.add_edge(agent1, agent2)

        reversed_backend = workflow.graph_backend.reverse()

        # In reversed graph, Agent2 should have Agent1 as predecessor
        preds = list(reversed_backend.predecessors("Agent1"))
        assert "Agent2" in preds

        # Agent2 should have no predecessors in reversed graph
        preds = list(reversed_backend.predecessors("Agent2"))
        assert len(preds) == 0

    def test_rustworkx_entry_end_points(self):
        """Test entry and end point detection"""
        workflow = GraphWorkflow(
            name="EntryEnd-Test", backend="rustworkx"
        )
        agent1 = create_test_agent("Agent1", "Entry agent")
        agent2 = create_test_agent("Agent2", "Middle agent")
        agent3 = create_test_agent("Agent3", "End agent")

        workflow.add_node(agent1)
        workflow.add_node(agent2)
        workflow.add_node(agent3)

        workflow.add_edge(agent1, agent2)
        workflow.add_edge(agent2, agent3)

        workflow.auto_set_entry_points()
        workflow.auto_set_end_points()

        assert "Agent1" in workflow.entry_points
        assert "Agent3" in workflow.end_points
        assert workflow.graph_backend.in_degree("Agent1") == 0
        assert workflow.graph_backend.out_degree("Agent3") == 0

    def test_rustworkx_isolated_nodes(self):
        """Test handling of isolated nodes"""
        workflow = GraphWorkflow(
            name="Isolated-Test", backend="rustworkx"
        )
        agent1 = create_test_agent("Agent1", "Connected agent")
        agent2 = create_test_agent("Agent2", "Isolated agent")

        workflow.add_node(agent1)
        workflow.add_node(agent2)
        workflow.add_edge(agent1, agent1)  # Self-loop

        workflow.compile()

        assert len(workflow.nodes) == 2
        assert "Agent2" in workflow.nodes

    def test_rustworkx_workflow_execution(self):
        """Test full workflow execution with rustworkx"""
        workflow = GraphWorkflow(
            name="Execution-Test", backend="rustworkx"
        )
        agent1 = create_test_agent("Agent1", "First agent")
        agent2 = create_test_agent("Agent2", "Second agent")

        workflow.add_node(agent1)
        workflow.add_node(agent2)
        workflow.add_edge(agent1, agent2)

        result = workflow.run("Test task")

        assert result is not None
        assert "Agent1" in result
        assert "Agent2" in result

    def test_rustworkx_compilation_caching(self):
        """Test that compilation is cached correctly"""
        workflow = GraphWorkflow(
            name="Cache-Test", backend="rustworkx"
        )
        agent1 = create_test_agent("Agent1", "First agent")
        agent2 = create_test_agent("Agent2", "Second agent")

        workflow.add_node(agent1)
        workflow.add_node(agent2)
        workflow.add_edge(agent1, agent2)

        # First compilation
        workflow.compile()
        layers1 = workflow._sorted_layers.copy()
        compiled1 = workflow._compiled

        # Second compilation should use cache
        workflow.compile()
        layers2 = workflow._sorted_layers.copy()
        compiled2 = workflow._compiled

        assert compiled1 == compiled2 == True
        assert layers1 == layers2

    def test_rustworkx_node_metadata(self):
        """Test node metadata handling"""
        workflow = GraphWorkflow(
            name="Metadata-Test", backend="rustworkx"
        )
        agent = create_test_agent("Agent", "Test agent")

        workflow.add_node(
            agent, metadata={"priority": "high", "timeout": 60}
        )

        node_index = workflow.graph_backend._node_id_to_index["Agent"]
        node_data = workflow.graph_backend.graph[node_index]

        assert isinstance(node_data, dict)
        assert node_data.get("node_id") == "Agent"
        assert node_data.get("priority") == "high"
        assert node_data.get("timeout") == 60

    def test_rustworkx_edge_metadata(self):
        """Test edge metadata handling"""
        workflow = GraphWorkflow(
            name="EdgeMetadata-Test", backend="rustworkx"
        )
        agent1 = create_test_agent("Agent1", "First agent")
        agent2 = create_test_agent("Agent2", "Second agent")

        workflow.add_node(agent1)
        workflow.add_node(agent2)
        workflow.add_edge(agent1, agent2, weight=5, label="test")

        assert len(workflow.edges) == 1
        assert workflow.edges[0].metadata.get("weight") == 5
        assert workflow.edges[0].metadata.get("label") == "test"


@pytest.mark.skipif(
    not RUSTWORKX_AVAILABLE, reason="rustworkx not available"
)
class TestRustworkxPerformance:
    """Performance tests for rustworkx backend"""

    def test_rustworkx_large_graph_compilation(self):
        """Test compilation performance with large graph"""
        workflow = GraphWorkflow(
            name="LargeGraph-Test", backend="rustworkx"
        )
        agents = [
            create_test_agent(f"Agent{i}", f"Agent {i}")
            for i in range(50)
        ]

        for agent in agents:
            workflow.add_node(agent)

        # Create a complex topology
        for i in range(len(agents) - 1):
            workflow.add_edge(agents[i], agents[i + 1])

        import time

        start = time.time()
        workflow.compile()
        compile_time = time.time() - start

        assert compile_time < 1.0  # Should compile quickly
        assert len(workflow._sorted_layers) == 50

    def test_rustworkx_many_predecessors(self):
        """Test performance with many predecessors"""
        workflow = GraphWorkflow(
            name="ManyPreds-Test", backend="rustworkx"
        )
        target = create_test_agent("Target", "Target agent")
        sources = [
            create_test_agent(f"Source{i}", f"Source {i}")
            for i in range(100)
        ]

        workflow.add_node(target)
        for source in sources:
            workflow.add_node(source)

        workflow.add_edges_to_target(sources, target)

        workflow.compile()

        predecessors = list(
            workflow.graph_backend.predecessors("Target")
        )
        assert len(predecessors) == 100


@pytest.mark.skipif(
    not RUSTWORKX_AVAILABLE, reason="rustworkx not available"
)
class TestRustworkxEdgeCases:
    """Edge case tests for rustworkx backend"""

    def test_rustworkx_empty_graph(self):
        """Test empty graph handling"""
        workflow = GraphWorkflow(
            name="Empty-Test", backend="rustworkx"
        )
        workflow.compile()

        assert len(workflow._sorted_layers) == 0
        assert len(workflow.nodes) == 0

    def test_rustworkx_single_node(self):
        """Test single node graph"""
        workflow = GraphWorkflow(
            name="Single-Test", backend="rustworkx"
        )
        agent = create_test_agent("Agent", "Single agent")

        workflow.add_node(agent)
        workflow.compile()

        assert len(workflow._sorted_layers) == 1
        assert workflow._sorted_layers[0] == ["Agent"]

    def test_rustworkx_self_loop(self):
        """Test self-loop handling"""
        workflow = GraphWorkflow(
            name="SelfLoop-Test", backend="rustworkx"
        )
        agent = create_test_agent("Agent", "Self-looping agent")

        workflow.add_node(agent)
        workflow.add_edge(agent, agent)

        workflow.compile()

        assert len(workflow.edges) == 1
        assert workflow.graph_backend.in_degree("Agent") == 1
        assert workflow.graph_backend.out_degree("Agent") == 1

    def test_rustworkx_duplicate_edge(self):
        """Test duplicate edge handling"""
        workflow = GraphWorkflow(
            name="Duplicate-Test", backend="rustworkx"
        )
        agent1 = create_test_agent("Agent1", "First agent")
        agent2 = create_test_agent("Agent2", "Second agent")

        workflow.add_node(agent1)
        workflow.add_node(agent2)

        # Add same edge twice
        workflow.add_edge(agent1, agent2)
        workflow.add_edge(agent1, agent2)

        # rustworkx should handle duplicate edges
        assert (
            len(workflow.edges) == 2
        )  # Both edges are stored in workflow
        workflow.compile()  # Should not crash


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
