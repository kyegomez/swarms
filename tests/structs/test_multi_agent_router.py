import os

import time

import pytest

from swarms.structs.agent import Agent
from swarms.structs.multi_agent_router import MultiAgentRouter


def _minimal_agents():
    """Lightweight agents for tests that don't need a real LLM call."""
    return [
        Agent(
            agent_name="Agent1",
            agent_description="Test Agent1",
            system_prompt="You are Agent1",
            model_name="openai/gpt-4o",
            max_loops=1,
            verbose=False,
            print_on=False,
        ),
        Agent(
            agent_name="Agent2",
            agent_description="Test Agent2",
            system_prompt="You are Agent2",
            model_name="openai/gpt-4o",
            max_loops=1,
            verbose=False,
            print_on=False,
        ),
    ]


# Test fixtures
def real_agents():
    """Create real agents for testing"""
    return [
        Agent(
            agent_name="ResearchAgent",
            agent_description="Specializes in researching topics and providing detailed, factual information",
            system_prompt="You are a research specialist. Provide detailed, well-researched information about any topic, citing sources when possible.",
            model_name="gpt-5.4",
            max_loops=1,
            verbose=False,
            print_on=False,
        ),
        Agent(
            agent_name="CodeExpertAgent",
            agent_description="Expert in writing, reviewing, and explaining code across multiple programming languages",
            system_prompt="You are a coding expert. Write, review, and explain code with a focus on best practices and clean code principles.",
            model_name="gpt-5.4",
            max_loops=1,
            verbose=False,
            print_on=False,
        ),
        Agent(
            agent_name="WritingAgent",
            agent_description="Skilled in creative and technical writing, content creation, and editing",
            system_prompt="You are a writing specialist. Create, edit, and improve written content while maintaining appropriate tone and style.",
            model_name="gpt-5.4",
            max_loops=1,
            verbose=False,
            print_on=False,
        ),
        Agent(
            agent_name="MathAgent",
            agent_description="Expert in mathematical calculations and problem solving",
            system_prompt="You are a math expert. Solve mathematical problems and explain solutions clearly.",
            model_name="gpt-5.4",
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
    assert router.model == "gpt-5.4"
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


# ============================================================================
# BOSS SYSTEM PROMPT / DISCOVERY TESTS
# ============================================================================


def test_boss_system_prompt_contains_agent_names():
    """The generated boss prompt must mention each registered agent by name."""
    router = MultiAgentRouter(agents=_minimal_agents())
    prompt = router._create_boss_system_prompt()

    assert "Agent1" in prompt
    assert "Agent2" in prompt
    assert "You are a boss agent" in prompt


def test_agents_dict_membership():
    """router.agents is a name-keyed dict; lookups should be membership tests."""
    router = MultiAgentRouter(agents=_minimal_agents())
    assert "Agent1" in router.agents
    assert "NonexistentAgent" not in router.agents


# ============================================================================
# CONSTRUCTION-TIME VALIDATION
# ============================================================================


def test_missing_api_key_raises():
    """MultiAgentRouter with no agents and no OPENAI_API_KEY must raise ValueError."""
    saved = os.environ.pop("OPENAI_API_KEY", None)
    try:
        with pytest.raises(ValueError, match="OpenAI API key"):
            MultiAgentRouter(agents=[])
    finally:
        if saved is not None:
            os.environ["OPENAI_API_KEY"] = saved


def test_route_task_with_no_agents_raises():
    """Routing with an empty agent list must raise."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    router = MultiAgentRouter(agents=[])
    with pytest.raises(Exception):
        router.route_task("Test task")


def test_route_task_with_empty_task_raises():
    """An empty task string must raise ValueError."""
    router = MultiAgentRouter(agents=_minimal_agents())
    with pytest.raises(ValueError):
        router.route_task("")


def _local_router(agents=None, delay=0.0):
    """A router whose agents answer locally, so no model is contacted."""
    from swarms.structs.conversation import Conversation

    ran = []

    class LocalAgent(Agent):
        def __init__(self, name):
            self.agent_name = name
            self.description = name

        def run(self, task, *args, **kwargs):
            if delay:
                time.sleep(delay)
            ran.append((self.agent_name, task))
            return f"{self.agent_name}-out"

    router = MultiAgentRouter.__new__(MultiAgentRouter)
    router.agents = [LocalAgent(n) for n in (agents or ["A1", "A2"])]
    router.skip_null_tasks = True
    router.print_on = False
    router.conversation = Conversation()
    router.output_type = "dict"
    return router, ran


def test_skip_null_tasks_actually_skips_on_the_single_path():
    """The single path logged "Skipping" and then ran the agent anyway."""
    router, ran = _local_router()

    router.handle_single_handoff(
        {"handoffs": [{"agent_name": "A1", "task": None}]}, ""
    )

    assert ran == [], f"the agent ran despite a null task: {ran}"


def test_a_handoff_without_a_task_key_falls_back_to_the_original():
    """HandOffsResponse.task is Optional, so the key may be absent."""
    router, ran = _local_router()

    router.handle_single_handoff(
        {"handoffs": [{"agent_name": "A1"}]}, "the original task"
    )

    assert ran == [("A1", "the original task")]


def test_multiple_handoffs_without_a_task_key_fall_back():
    router, ran = _local_router()

    router.handle_multiple_handoffs(
        {"handoffs": [{"agent_name": "A1"}, {"agent_name": "A2"}]},
        "the original task",
    )

    assert [task for _, task in ran] == [
        "the original task",
        "the original task",
    ]


def test_selected_agents_run_concurrently():
    """Independent agents used to run one after another."""
    router, ran = _local_router(agents=["S0", "S1", "S2"], delay=0.35)
    handoffs = {
        "handoffs": [
            {"agent_name": f"S{i}", "task": f"t{i}"} for i in range(3)
        ]
    }

    started = time.time()
    router.handle_multiple_handoffs(handoffs, "x")
    elapsed = time.time() - started

    assert len(ran) == 3
    assert (
        elapsed < 0.35 * 3 * 0.7
    ), f"three 0.35s agents took {elapsed:.2f}s, so they ran in series"


def test_every_selected_agent_answer_reaches_the_conversation():
    """Only agent_responses[0] was recorded; the rest ran, billed, and were dropped."""
    router, ran = _local_router(agents=["A1", "A2", "A3"])

    router.handle_multiple_handoffs(
        {
            "handoffs": [
                {"agent_name": n, "task": "t"}
                for n in ("A1", "A2", "A3")
            ]
        },
        "x",
    )

    assert len(ran) == 3
    assert [
        (m["role"], m["content"])
        for m in router.conversation.conversation_history
    ] == [("A1", "A1-out"), ("A2", "A2-out"), ("A3", "A3-out")]


def test_an_unknown_agent_raises_before_any_agent_runs():
    router, ran = _local_router()

    with pytest.raises(ValueError, match="unknown agent"):
        router.handle_multiple_handoffs(
            {
                "handoffs": [
                    {"agent_name": "A1", "task": "a"},
                    {"agent_name": "NOPE", "task": "b"},
                ]
            },
            "x",
        )

    assert ran == [], f"an agent ran before validation failed: {ran}"


def test_concurrent_batch_run_returns_results_in_input_order():
    """Results used to come back in completion order, unlike batch_run."""
    router, _ = _local_router()

    def flaky(task):
        if task == "bad":
            raise RuntimeError("boom")
        return f"ok:{task}"

    router.route_task = flaky

    assert router.concurrent_batch_run(["a", "bad", "c"]) == [
        "ok:a",
        "ok:c",
    ]


if __name__ == "__main__":
    pytest.main([__file__])
