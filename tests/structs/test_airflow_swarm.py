import time

from loguru import logger
from swarms import Agent

from swarms.structs.airflow_swarm import (
    AirflowDAGSwarm,
    NodeType,
    Conversation,
)

# Configure logger
logger.remove()
logger.add(lambda msg: print(msg, end=""), level="DEBUG")


def test_swarm_initialization():
    """Test basic swarm initialization and configuration."""
    try:
        swarm = AirflowDAGSwarm(
            dag_id="test_dag",
            name="Test DAG",
            initial_message="Test message",
        )
        assert swarm.dag_id == "test_dag", "DAG ID not set correctly"
        assert swarm.name == "Test DAG", "Name not set correctly"
        assert (
            len(swarm.nodes) == 0
        ), "Nodes should be empty on initialization"
        assert (
            len(swarm.edges) == 0
        ), "Edges should be empty on initialization"

        # Test initial message
        conv_json = swarm.get_conversation_history()
        assert (
            "Test message" in conv_json
        ), "Initial message not set correctly"
        print("✅ Swarm initialization test passed")
        return True
    except AssertionError as e:
        print(f"❌ Swarm initialization test failed: {str(e)}")
        return False


def test_node_addition():
    """Test adding different types of nodes to the swarm."""
    try:
        swarm = AirflowDAGSwarm(dag_id="test_dag")

        # Test adding an agent node
        agent = Agent(
            agent_name="Test-Agent",
            system_prompt="Test prompt",
            model_name="gpt-4o-mini",
            max_loops=1,
        )
        agent_id = swarm.add_node(
            "test_agent",
            agent,
            NodeType.AGENT,
            query="Test query",
            concurrent=True,
        )
        assert (
            agent_id == "test_agent"
        ), "Agent node ID not returned correctly"
        assert (
            "test_agent" in swarm.nodes
        ), "Agent node not added to nodes dict"

        # Test adding a callable node
        def test_callable(x: int, conversation: Conversation) -> str:
            return f"Test output {x}"

        callable_id = swarm.add_node(
            "test_callable",
            test_callable,
            NodeType.CALLABLE,
            args=[42],
            concurrent=False,
        )
        assert (
            callable_id == "test_callable"
        ), "Callable node ID not returned correctly"
        assert (
            "test_callable" in swarm.nodes
        ), "Callable node not added to nodes dict"

        print("✅ Node addition test passed")
        return True
    except AssertionError as e:
        print(f"❌ Node addition test failed: {str(e)}")
        return False
    except Exception as e:
        print(
            f"❌ Node addition test failed with unexpected error: {str(e)}"
        )
        return False


def test_edge_addition():
    """Test adding edges between nodes."""
    try:
        swarm = AirflowDAGSwarm(dag_id="test_dag")

        # Add two nodes
        def node1_fn(conversation: Conversation) -> str:
            return "Node 1 output"

        def node2_fn(conversation: Conversation) -> str:
            return "Node 2 output"

        swarm.add_node("node1", node1_fn, NodeType.CALLABLE)
        swarm.add_node("node2", node2_fn, NodeType.CALLABLE)

        # Add edge between them
        swarm.add_edge("node1", "node2")

        assert (
            "node2" in swarm.edges["node1"]
        ), "Edge not added correctly"
        assert (
            len(swarm.edges["node1"]) == 1
        ), "Incorrect number of edges"

        # Test adding edge with non-existent node
        try:
            swarm.add_edge("node1", "non_existent")
            assert (
                False
            ), "Should raise ValueError for non-existent node"
        except ValueError:
            pass

        print("✅ Edge addition test passed")
        return True
    except AssertionError as e:
        print(f"❌ Edge addition test failed: {str(e)}")
        return False


def test_execution_order():
    """Test that nodes are executed in the correct order based on dependencies."""
    try:
        swarm = AirflowDAGSwarm(dag_id="test_dag")
        execution_order = []

        def node1(conversation: Conversation) -> str:
            execution_order.append("node1")
            return "Node 1 output"

        def node2(conversation: Conversation) -> str:
            execution_order.append("node2")
            return "Node 2 output"

        def node3(conversation: Conversation) -> str:
            execution_order.append("node3")
            return "Node 3 output"

        # Add nodes
        swarm.add_node(
            "node1", node1, NodeType.CALLABLE, concurrent=False
        )
        swarm.add_node(
            "node2", node2, NodeType.CALLABLE, concurrent=False
        )
        swarm.add_node(
            "node3", node3, NodeType.CALLABLE, concurrent=False
        )

        # Add edges to create a chain: node1 -> node2 -> node3
        swarm.add_edge("node1", "node2")
        swarm.add_edge("node2", "node3")

        # Execute
        swarm.run()

        # Check execution order
        assert execution_order == [
            "node1",
            "node2",
            "node3",
        ], "Incorrect execution order"
        print("✅ Execution order test passed")
        return True
    except AssertionError as e:
        print(f"❌ Execution order test failed: {str(e)}")
        return False


def test_concurrent_execution():
    """Test concurrent execution of nodes."""
    try:
        swarm = AirflowDAGSwarm(dag_id="test_dag")

        def slow_node1(conversation: Conversation) -> str:
            time.sleep(0.5)
            return "Slow node 1 output"

        def slow_node2(conversation: Conversation) -> str:
            time.sleep(0.5)
            return "Slow node 2 output"

        # Add nodes with concurrent=True
        swarm.add_node(
            "slow1", slow_node1, NodeType.CALLABLE, concurrent=True
        )
        swarm.add_node(
            "slow2", slow_node2, NodeType.CALLABLE, concurrent=True
        )

        # Measure execution time
        start_time = time.time()
        swarm.run()
        execution_time = time.time() - start_time

        # Should take ~0.5s for concurrent execution, not ~1s
        assert (
            execution_time < 0.8
        ), "Concurrent execution took too long"
        print("✅ Concurrent execution test passed")
        return True
    except AssertionError as e:
        print(f"❌ Concurrent execution test failed: {str(e)}")
        return False


def test_conversation_handling():
    """Test conversation management within the swarm."""
    try:
        swarm = AirflowDAGSwarm(
            dag_id="test_dag", initial_message="Initial test message"
        )

        # Test adding user messages
        swarm.add_user_message("Test message 1")
        swarm.add_user_message("Test message 2")

        history = swarm.get_conversation_history()
        assert (
            "Initial test message" in history
        ), "Initial message not in history"
        assert (
            "Test message 1" in history
        ), "First message not in history"
        assert (
            "Test message 2" in history
        ), "Second message not in history"

        print("✅ Conversation handling test passed")
        return True
    except AssertionError as e:
        print(f"❌ Conversation handling test failed: {str(e)}")
        return False


def test_error_handling():
    """Test error handling in node execution."""
    try:
        swarm = AirflowDAGSwarm(dag_id="test_dag")

        def failing_node(conversation: Conversation) -> str:
            raise ValueError("Test error")

        swarm.add_node("failing", failing_node, NodeType.CALLABLE)

        # Execute should not raise an exception
        result = swarm.run()

        assert (
            "Error" in result
        ), "Error not captured in execution result"
        assert (
            "Test error" in result
        ), "Specific error message not captured"

        print("✅ Error handling test passed")
        return True
    except Exception as e:
        print(f"❌ Error handling test failed: {str(e)}")
        return False


def run_all_tests():
    """Run all test functions and report results."""
    tests = [
        test_swarm_initialization,
        test_node_addition,
        test_edge_addition,
        test_execution_order,
        test_concurrent_execution,
        test_conversation_handling,
        test_error_handling,
    ]

    results = []
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        result = test()
        results.append(result)

    total = len(results)
    passed = sum(results)
    print("\n=== Test Results ===")
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print("==================")


if __name__ == "__main__":
    run_all_tests()
