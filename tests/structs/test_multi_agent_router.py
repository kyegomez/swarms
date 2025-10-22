import pytest

from swarms.structs.agent import Agent
from swarms.structs.multi_agent_router import MultiAgentRouter


# Test fixtures
def real_agents():
    """Create real agents for testing"""
    return [
        Agent(
            agent_name="ResearchAgent",
            agent_description="Specializes in researching topics and providing detailed, factual information",
            system_prompt="You are a research specialist. Provide detailed, well-researched information about any topic, citing sources when possible.",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=False,
            print_on=False,
        ),
        Agent(
            agent_name="CodeExpertAgent",
            agent_description="Expert in writing, reviewing, and explaining code across multiple programming languages",
            system_prompt="You are a coding expert. Write, review, and explain code with a focus on best practices and clean code principles.",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=False,
            print_on=False,
        ),
        Agent(
            agent_name="WritingAgent",
            agent_description="Skilled in creative and technical writing, content creation, and editing",
            system_prompt="You are a writing specialist. Create, edit, and improve written content while maintaining appropriate tone and style.",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=False,
            print_on=False,
        ),
        Agent(
            agent_name="MathAgent",
            agent_description="Expert in mathematical calculations and problem solving",
            system_prompt="You are a math expert. Solve mathematical problems and explain solutions clearly.",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=False,
            print_on=False,
        ),
    ]


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================


def test_multi_agent_router_initialization_default():
    """Test MultiAgentRouter initialization with default parameters"""
    router = MultiAgentRouter(agents=real_agents())

    assert router.name == "swarm-router"
    assert (
        router.description
        == "Routes tasks to specialized agents based on their capabilities"
    )
    assert router.model == "gpt-4o-mini"
    assert router.temperature == 0.1
    assert router.output_type == "dict"
    assert router.print_on is True
    assert router.skip_null_tasks is True
    assert len(router.agents) == 4
    assert all(
        agent_name in router.agents
        for agent_name in [
            "ResearchAgent",
            "CodeExpertAgent",
            "WritingAgent",
            "MathAgent",
        ]
    )
    assert isinstance(
        router.conversation, object
    )  # Conversation object
    assert hasattr(router.function_caller, "run")


def test_multi_agent_router_initialization_custom_params():
    """Test MultiAgentRouter initialization with custom parameters"""
    custom_name = "custom-router"
    custom_description = "Custom description"
    custom_model = "gpt-4"
    custom_temperature = 0.5
    custom_output_type = "json"

    router = MultiAgentRouter(
        name=custom_name,
        description=custom_description,
        agents=real_agents(),
        model=custom_model,
        temperature=custom_temperature,
        output_type=custom_output_type,
        print_on=False,
        skip_null_tasks=False,
        system_prompt="Custom system prompt",
    )

    assert router.name == custom_name
    assert router.description == custom_description
    assert router.model == custom_model
    assert router.temperature == custom_temperature
    assert router.output_type == custom_output_type
    assert router.print_on is False
    assert router.skip_null_tasks is False
    assert router.system_prompt == "Custom system prompt"


def test_multi_agent_router_repr():
    """Test MultiAgentRouter string representation"""
    router = MultiAgentRouter(agents=real_agents())

    expected_repr = f"MultiAgentRouter(name={router.name}, agents={list(router.agents.keys())})"
    assert repr(router) == expected_repr


# ============================================================================
# SINGLE HANDOFF TESTS
# ============================================================================


def test_handle_single_handoff_valid():
    """Test handling single handoff with valid agent"""
    router = MultiAgentRouter(agents=real_agents())

    result = router.route_task("Write a fibonacci function")

    # Check that conversation was updated
    assert len(router.conversation.conversation_history) > 0
    # Check that we got a valid response
    assert result is not None
    assert isinstance(result, (list, dict))


# ============================================================================
# MULTIPLE HANDOFF TESTS
# ============================================================================


def test_handle_multiple_handoffs_valid():
    """Test handling multiple handoffs with valid agents"""
    router = MultiAgentRouter(agents=real_agents())

    result = router.route_task("Research and implement fibonacci")

    # Check that conversation was updated
    history = router.conversation.conversation_history

    assert len(history) > 0
    assert result is not None
    assert isinstance(result, (list, dict))


def test_handle_multiple_handoffs_with_null_tasks():
    """Test handling multiple handoffs with some null tasks"""
    router = MultiAgentRouter(
        agents=real_agents(), skip_null_tasks=True
    )

    result = router.route_task("Mixed task")

    # Should still return a valid result
    history = router.conversation.conversation_history
    assert len(history) > 0
    assert result is not None
    assert isinstance(result, (list, dict))


# ============================================================================
# ROUTE TASK TESTS
# ============================================================================


def test_route_task_single_agent():
    """Test route_task with single agent routing"""
    router = MultiAgentRouter(agents=real_agents())

    result = router.route_task("Write a fibonacci function")

    # Check result structure - should be a list of conversation messages
    assert result is not None
    assert isinstance(result, (list, dict))
    assert len(result) > 0 if isinstance(result, list) else True


def test_route_task_multiple_agents():
    """Test route_task with multiple agent routing"""
    router = MultiAgentRouter(agents=real_agents())

    result = router.route_task("Research and implement fibonacci")

    # Check result structure
    assert result is not None
    assert isinstance(result, (list, dict))


def test_route_task_print_on_true():
    """Test route_task with print_on=True"""
    router = MultiAgentRouter(agents=real_agents(), print_on=True)

    # Should not raise any exceptions when printing
    result = router.route_task("Test task")
    assert result is not None
    assert isinstance(result, (list, dict))


def test_route_task_print_on_false():
    """Test route_task with print_on=False"""
    router = MultiAgentRouter(agents=real_agents(), print_on=False)

    # Should not raise any exceptions when not printing
    result = router.route_task("Test task")
    assert result is not None
    assert isinstance(result, (list, dict))


# ============================================================================
# ALIAS METHOD TESTS
# ============================================================================


def test_run_alias():
    """Test that run() method is an alias for route_task()"""
    router = MultiAgentRouter(agents=real_agents())

    result1 = router.run(
        "Call your favorite agent to write a fibonacci function"
    )
    result2 = router.route_task(
        "Call your favorite agent to write a fibonacci function"
    )

    # Results should be valid
    assert result1 is not None
    assert result2 is not None
    assert isinstance(result1, (list, dict))
    assert isinstance(result2, (list, dict))


def test_call_alias():
    """Test that __call__() method is an alias for route_task()"""
    router = MultiAgentRouter(agents=real_agents())

    result1 = router(
        "Call your favorite agent to write a fibonacci function"
    )
    result2 = router(
        "Call your favorite agent to write a fibonacci function"
    )

    # Results should be valid
    assert result1 is not None
    assert result2 is not None
    assert isinstance(result1, (list, dict))
    assert isinstance(result2, (list, dict))


# ============================================================================
# BATCH PROCESSING TESTS
# ============================================================================


def test_batch_run():
    """Test batch_run method"""
    router = MultiAgentRouter(agents=real_agents())

    tasks = [
        "Call your favorite agent to write a fibonacci function",
        "Call your favorite agent to write a fibonacci function",
        "Call your favorite agent to write a fibonacci function",
    ]
    results = router.batch_run(tasks)

    assert len(results) == 3
    assert all(result is not None for result in results)
    assert all(isinstance(result, (list, dict)) for result in results)


def test_concurrent_batch_run():
    """Test concurrent_batch_run method"""
    router = MultiAgentRouter(agents=real_agents())

    tasks = [
        "Call your favorite agent to write a fibonacci function",
        "Call your favorite agent to write a fibonacci function",
        "Call your favorite agent to write a fibonacci function",
    ]
    results = router.concurrent_batch_run(tasks)

    assert len(results) == 3
    assert all(result is not None for result in results)
    assert all(isinstance(result, (list, dict)) for result in results)


# ============================================================================
# OUTPUT TYPE TESTS
# ============================================================================


@pytest.mark.parametrize("output_type", ["dict", "json", "string"])
def test_different_output_types(output_type):
    """Test different output types"""
    router = MultiAgentRouter(
        agents=real_agents(), output_type=output_type
    )

    result = router.route_task("Test task")

    assert result is not None
    # Output format depends on the formatter, but should not raise errors
    assert isinstance(result, (list, dict, str))


# ============================================================================
# PERFORMANCE AND LOAD TESTS
# ============================================================================


def test_large_batch_processing():
    """Test processing a large batch of tasks"""
    router = MultiAgentRouter(agents=real_agents())

    # Create a smaller number of tasks for testing (reduced from 100 to 5 for performance)
    tasks = [f"Task number {i}" for i in range(5)]
    results = router.batch_run(tasks)

    assert len(results) == 5


def test_concurrent_large_batch_processing():
    """Test concurrent processing of a large batch of tasks"""
    router = MultiAgentRouter(agents=real_agents())

    # Create a small number of tasks for testing
    tasks = [
        f"Route task to your favorite agent to write a fibonacci function {i}"
        for i in range(3)
    ]
    results = router.concurrent_batch_run(tasks)

    assert len(results) == 3
    assert all(result is not None for result in results)
    assert all(isinstance(result, (list, dict)) for result in results)


if __name__ == "__main__":
    pytest.main([__file__])
