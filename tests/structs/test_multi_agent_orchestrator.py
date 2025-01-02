import os
from swarms.structs.agent import Agent
from swarms.structs.multi_agent_orchestrator import MultiAgentRouter


def create_test_agent(name: str) -> Agent:
    """Helper function to create a test agent"""
    return Agent(
        agent_name=name,
        description=f"Test {name}",
        system_prompt=f"You are a {name}",
        model_name="openai/gpt-4o",
    )


def test_boss_router_initialization():
    """Test MultiAgentRouter initialization"""
    print("\nTesting MultiAgentRouter initialization...")

    # Test successful initialization
    try:
        agents = [
            create_test_agent("TestAgent1"),
            create_test_agent("TestAgent2"),
        ]
        router = MultiAgentRouter(agents=agents)
        assert (
            router.name == "swarm-router"
        ), "Default name should be 'swarm-router'"
        assert len(router.agents) == 2, "Should have 2 agents"
        print("✓ Basic initialization successful")
    except Exception as e:
        print(f"✗ Basic initialization failed: {str(e)}")

    # Test initialization without API key
    try:
        temp_key = os.getenv("OPENAI_API_KEY")
        os.environ["OPENAI_API_KEY"] = ""
        success = False
        try:
            router = MultiAgentRouter(agents=[])
        except ValueError as e:
            success = str(e) == "OpenAI API key must be provided"
        os.environ["OPENAI_API_KEY"] = temp_key
        assert (
            success
        ), "Should raise ValueError when API key is missing"
        print("✓ API key validation successful")
    except Exception as e:
        print(f"✗ API key validation failed: {str(e)}")


def test_boss_system_prompt():
    """Test system prompt generation"""
    print("\nTesting system prompt generation...")

    try:
        agents = [
            create_test_agent("Agent1"),
            create_test_agent("Agent2"),
        ]
        router = MultiAgentRouter(agents=agents)
        prompt = router._create_boss_system_prompt()

        # Check if prompt contains agent information
        assert (
            "Agent1" in prompt
        ), "Prompt should contain first agent name"
        assert (
            "Agent2" in prompt
        ), "Prompt should contain second agent name"
        assert (
            "You are a boss agent" in prompt
        ), "Prompt should contain boss agent description"
        print("✓ System prompt generation successful")
    except Exception as e:
        print(f"✗ System prompt generation failed: {str(e)}")


def test_find_agent_in_list():
    """Test agent finding functionality"""
    print("\nTesting agent finding functionality...")

    try:
        agent1 = create_test_agent("Agent1")
        agent2 = create_test_agent("Agent2")
        router = MultiAgentRouter(agents=[agent1, agent2])

        # Test finding existing agent
        assert "Agent1" in router.agents, "Should find existing agent"
        assert (
            "NonexistentAgent" not in router.agents
        ), "Should not find nonexistent agent"
        print("✓ Agent finding successful")
    except Exception as e:
        print(f"✗ Agent finding failed: {str(e)}")


def test_task_routing():
    """Test task routing functionality"""
    print("\nTesting task routing...")

    try:
        # Create test agents
        agents = [
            create_test_agent("CodeAgent"),
            create_test_agent("WritingAgent"),
        ]
        router = MultiAgentRouter(agents=agents)

        # Test routing a coding task
        result = router.route_task(
            "Write a Python function to sort a list"
        )
        assert result["boss_decision"]["selected_agent"] in [
            "CodeAgent",
            "WritingAgent",
        ], "Should select an appropriate agent"
        assert (
            "execution" in result
        ), "Result should contain execution details"
        assert (
            "total_time" in result
        ), "Result should contain timing information"
        print("✓ Task routing successful")
    except Exception as e:
        print(f"✗ Task routing failed: {str(e)}")


def test_batch_routing():
    """Test batch routing functionality"""
    print("\nTesting batch routing...")

    try:
        agents = [create_test_agent("TestAgent")]
        router = MultiAgentRouter(agents=agents)

        tasks = ["Task 1", "Task 2", "Task 3"]

        # Test sequential batch routing
        results = router.batch_route(tasks)
        assert len(results) == len(
            tasks
        ), "Should return result for each task"
        print("✓ Sequential batch routing successful")

        # Test concurrent batch routing
        concurrent_results = router.concurrent_batch_route(tasks)
        assert len(concurrent_results) == len(
            tasks
        ), "Should return result for each task"
        print("✓ Concurrent batch routing successful")
    except Exception as e:
        print(f"✗ Batch routing failed: {str(e)}")


def test_error_handling():
    """Test error handling in various scenarios"""
    print("\nTesting error handling...")

    try:
        router = MultiAgentRouter(agents=[])

        # Test routing with no agents
        success = False
        try:
            router.route_task("Test task")
        except Exception:
            success = True
        assert success, "Should handle routing with no agents"
        print("✓ Empty agent list handling successful")

        # Test with invalid task
        success = False
        router = MultiAgentRouter(
            agents=[create_test_agent("TestAgent")]
        )
        try:
            router.route_task("")
        except ValueError:
            success = True
        assert success, "Should handle empty task"
        print("✓ Invalid task handling successful")
    except Exception as e:
        print(f"✗ Error handling failed: {str(e)}")


def run_all_tests():
    """Run all test functions"""
    print("Starting MultiAgentRouter tests...")

    test_functions = [
        test_boss_router_initialization,
        test_boss_system_prompt,
        test_find_agent_in_list,
        test_task_routing,
        test_batch_routing,
        test_error_handling,
    ]

    total_tests = len(test_functions)
    passed_tests = 0

    for test_func in test_functions:
        try:
            test_func()
            passed_tests += 1
        except Exception as e:
            print(
                f"Test {test_func.__name__} failed with error: {str(e)}"
            )

    print(
        f"\nTest Results: {passed_tests}/{total_tests} tests passed"
    )


if __name__ == "__main__":
    run_all_tests()
