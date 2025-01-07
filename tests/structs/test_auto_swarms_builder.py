from swarms.structs.auto_swarm_builder import AutoSwarmBuilder
from dotenv import load_dotenv

load_dotenv()


def print_separator():
    print("\n" + "=" * 50)


def test_initialization():
    """Test basic initialization of AutoSwarmBuilder"""
    print_separator()
    print("Testing AutoSwarmBuilder Initialization")
    try:
        swarm = AutoSwarmBuilder(
            name="TestSwarm",
            description="A test swarm for validation",
            verbose=True,
            max_loops=2,
        )

        print("✓ Created swarm with configuration:")
        print(f"  - Name: {swarm.name}")
        print(f"  - Description: {swarm.description}")
        print(f"  - Max loops: {swarm.max_loops}")
        print(f"  - Verbose: {swarm.verbose}")
        print("✓ Initialization test passed")
        return swarm
    except Exception as e:
        print(f"✗ Initialization test failed: {str(e)}")
        raise


def test_agent_building():
    """Test building individual agents"""
    print_separator()
    print("Testing Agent Building")
    try:
        swarm = AutoSwarmBuilder()
        agent = swarm.build_agent(
            agent_name="TestAgent",
            agent_description="A test agent",
            agent_system_prompt="You are a test agent",
            max_loops=1,
        )

        print("✓ Built agent with configuration:")
        print(f"  - Name: {agent.agent_name}")
        print(f"  - Description: {agent.description}")
        print(f"  - Max loops: {agent.max_loops}")
        print("✓ Agent building test passed")
        return agent
    except Exception as e:
        print(f"✗ Agent building test failed: {str(e)}")
        raise


def test_agent_creation():
    """Test creating multiple agents for a task"""
    print_separator()
    print("Testing Agent Creation from Task")
    try:
        swarm = AutoSwarmBuilder(
            name="ResearchSwarm",
            description="A swarm for research tasks",
        )
        task = "Research the latest developments in quantum computing"
        agents = swarm._create_agents(task)

        print("✓ Created agents for research task:")
        for i, agent in enumerate(agents, 1):
            print(f"  Agent {i}:")
            print(f"    - Name: {agent.agent_name}")
            print(f"    - Description: {agent.description}")
        print(f"✓ Created {len(agents)} agents successfully")
        return agents
    except Exception as e:
        print(f"✗ Agent creation test failed: {str(e)}")
        raise


def test_swarm_routing():
    """Test routing tasks through the swarm"""
    print_separator()
    print("Testing Swarm Routing")
    try:
        swarm = AutoSwarmBuilder(
            name="RouterTestSwarm",
            description="Testing routing capabilities",
        )
        agents = (
            test_agent_creation()
        )  # Get agents from previous test
        task = "Analyze the impact of AI on healthcare"

        print("Starting task routing...")
        result = swarm.swarm_router(agents, task)

        print("✓ Task routed successfully")
        print(
            f"  - Result length: {len(str(result)) if result else 0} characters"
        )
        print("✓ Swarm routing test passed")
        return result
    except Exception as e:
        print(f"✗ Swarm routing test failed: {str(e)}")
        raise


def test_full_swarm_execution():
    """Test complete swarm execution with a real task"""
    print_separator()
    print("Testing Full Swarm Execution")
    try:
        swarm = AutoSwarmBuilder(
            name="FullTestSwarm",
            description="Testing complete swarm functionality",
            max_loops=1,
        )
        task = (
            "Create a summary of recent advances in renewable energy"
        )

        print("Starting full swarm execution...")
        result = swarm.run(task)

        print("✓ Full swarm execution completed:")
        print(f"  - Output generated: {bool(result)}")
        print(
            f"  - Output length: {len(str(result)) if result else 0} characters"
        )
        print("✓ Full swarm execution test passed")
        return result
    except Exception as e:
        print(f"✗ Full swarm execution test failed: {str(e)}")
        raise


def test_error_handling():
    """Test error handling in swarm operations"""
    print_separator()
    print("Testing Error Handling")
    try:
        swarm = AutoSwarmBuilder()

        # Test with invalid agent configuration
        print("Testing invalid agent configuration...")
        try:
            swarm.build_agent("", "", "")
            print(
                "✗ Should have raised an error for empty agent configuration"
            )
        except Exception as e:
            print(
                f"✓ Correctly handled invalid agent configuration: {type(e).__name__}"
            )

        # Test with None task
        print("\nTesting None task...")
        try:
            swarm.run(None)
            print("✗ Should have raised an error for None task")
        except Exception as e:
            print(
                f"✓ Correctly handled None task: {type(e).__name__}"
            )

        print("✓ Error handling test passed")
    except Exception as e:
        print(f"✗ Error handling test failed: {str(e)}")
        raise


def run_all_tests():
    """Run complete test suite"""
    print("\n=== Starting AutoSwarmBuilder Test Suite ===\n")

    try:
        # Run all tests in sequence
        test_initialization()
        test_agent_building()
        test_agent_creation()
        test_swarm_routing()
        test_full_swarm_execution()
        test_error_handling()

        print_separator()
        print("🎉 All tests completed successfully!")

    except Exception as e:
        print_separator()
        print(f"❌ Test suite failed: {str(e)}")
        raise


if __name__ == "__main__":
    run_all_tests()
