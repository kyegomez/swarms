"""
Unit and integration tests for AgentRearrange.

Covers:
- Initialization, agent management, flow validation
- Sequential and concurrent flow execution
- Sequential awareness (including repeated agents)
- Error propagation through run/__call__/batch_run
- batch_run concurrency, ordering, conversation isolation, image forwarding,
  and batch_size validation (mock-based unit tests)
"""

import threading
import time
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from swarms import Agent, AgentRearrange


# ============================================================================
# Helper Functions
# ============================================================================


def create_sample_agents():
    """Create sample agents for integration tests against a real LLM."""
    return [
        Agent(
            agent_name="ResearchAgent",
            agent_description="Specializes in researching topics",
            system_prompt="You are a research specialist. Provide concise answers.",
            model_name="gpt-5.4",
            max_loops=1,
            verbose=True,
            streaming_on=True,
        ),
        Agent(
            agent_name="WriterAgent",
            agent_description="Expert in writing content",
            system_prompt="You are a writing expert. Provide concise answers.",
            model_name="gpt-5.4",
            max_loops=1,
            verbose=True,
            streaming_on=True,
        ),
        Agent(
            agent_name="ReviewerAgent",
            agent_description="Expert in reviewing content",
            system_prompt="You are a review expert. Provide concise answers.",
            model_name="gpt-5.4",
            max_loops=1,
            verbose=True,
            streaming_on=True,
        ),
    ]


def _make_agent(name: str, delay: float = 0.0):
    """Return a minimal mock Agent that sleeps *delay* seconds then echoes."""
    agent = MagicMock()
    agent.agent_name = name
    agent.system_prompt = f"I am {name}."

    def _run(task, *a, **kw):
        if delay:
            time.sleep(delay)
        return f"{name}:{task}"

    agent.run = _run
    return agent


def _make_pipeline(
    *agent_names: str, delay: float = 0.0
) -> AgentRearrange:
    """
    Build a sequential AgentRearrange pipeline with mock agents.

    Requires at least 2 agent names because validate_flow() demands '->'.
    """
    assert len(agent_names) >= 2, "_make_pipeline needs >=2 agents"
    agents = [_make_agent(n, delay=delay) for n in agent_names]
    flow = " -> ".join(agent_names)
    return AgentRearrange(
        agents=agents,
        flow=flow,
        max_loops=1,
        autosave=False,
        output_type="final",
    )


# ============================================================================
# Initialization Tests
# ============================================================================


def test_initialization():
    """Test AgentRearrange initialization."""
    agents = create_sample_agents()

    agent_rearrange = AgentRearrange(
        agents=agents,
        flow="ResearchAgent -> WriterAgent -> ReviewerAgent",
        max_loops=1,
        verbose=True,
    )

    assert len(agent_rearrange.agents) == 3
    assert (
        agent_rearrange.flow
        == "ResearchAgent -> WriterAgent -> ReviewerAgent"
    )
    assert agent_rearrange.max_loops == 1
    assert agent_rearrange.verbose is True


def test_initialization_with_team_awareness():
    """Test AgentRearrange with team_awareness enabled."""
    agents = create_sample_agents()

    agent_rearrange = AgentRearrange(
        agents=agents,
        flow="ResearchAgent -> WriterAgent -> ReviewerAgent",
        team_awareness=True,
        verbose=True,
    )

    assert (
        agent_rearrange.flow
        == "ResearchAgent -> WriterAgent -> ReviewerAgent"
    )


def test_initialization_with_custom_output_type():
    """Test AgentRearrange with custom output types."""
    agents = create_sample_agents()

    for output_type in ["all", "final", "list", "dict"]:
        agent_rearrange = AgentRearrange(
            agents=agents,
            flow="ResearchAgent -> WriterAgent",
            output_type=output_type,
            verbose=True,
        )
        assert agent_rearrange.output_type == output_type


# ============================================================================
# Agent Management Tests
# ============================================================================


def test_add_agent():
    """Test adding an agent to AgentRearrange."""
    agents = create_sample_agents()[:2]  # Only 2 agents

    agent_rearrange = AgentRearrange(
        agents=agents,
        flow="ResearchAgent -> WriterAgent",
        verbose=True,
    )

    new_agent = Agent(
        agent_name="EditorAgent",
        model_name="gpt-5.4",
        max_loops=1,
        verbose=True,
        streaming_on=True,
    )

    agent_rearrange.add_agent(new_agent)
    assert "EditorAgent" in agent_rearrange.agents
    assert len(agent_rearrange.agents) == 3


def test_remove_agent():
    """Test removing an agent from AgentRearrange."""
    agents = create_sample_agents()

    agent_rearrange = AgentRearrange(
        agents=agents,
        flow="ResearchAgent -> WriterAgent -> ReviewerAgent",
        verbose=True,
    )

    agent_rearrange.remove_agent("ReviewerAgent")
    assert "ReviewerAgent" not in agent_rearrange.agents
    assert len(agent_rearrange.agents) == 2


def test_add_agents():
    """Test adding multiple agents to AgentRearrange."""
    agents = create_sample_agents()[:1]  # Start with 1 agent

    agent_rearrange = AgentRearrange(
        agents=agents,
        flow="ResearchAgent",
        verbose=True,
    )

    new_agents = [
        Agent(
            agent_name="Agent4",
            model_name="gpt-5.4",
            max_loops=1,
            verbose=True,
            streaming_on=True,
        ),
        Agent(
            agent_name="Agent5",
            model_name="gpt-5.4",
            max_loops=1,
            verbose=True,
            streaming_on=True,
        ),
    ]

    agent_rearrange.add_agents(new_agents)
    assert "Agent4" in agent_rearrange.agents
    assert "Agent5" in agent_rearrange.agents
    assert len(agent_rearrange.agents) == 3


# ============================================================================
# Flow Validation Tests
# ============================================================================


def test_validate_flow_valid():
    """Test flow validation with valid flow."""
    agents = create_sample_agents()

    agent_rearrange = AgentRearrange(
        agents=agents,
        flow="ResearchAgent -> WriterAgent -> ReviewerAgent",
        verbose=True,
    )

    assert agent_rearrange.validate_flow() is True


def test_validate_flow_invalid():
    """Test flow validation with invalid agent name in flow."""
    agents = create_sample_agents()

    agent_rearrange = AgentRearrange(
        agents=agents,
        flow="ResearchAgent -> WriterAgent -> ReviewerAgent",
        verbose=True,
    )

    agent_rearrange.flow = "ResearchAgent -> NonExistentAgent"

    with pytest.raises(ValueError, match="not registered"):
        agent_rearrange.validate_flow()


def test_validate_flow_no_arrow():
    """Test flow validation without arrow syntax."""
    agents = create_sample_agents()

    agent_rearrange = AgentRearrange(
        agents=agents,
        flow="ResearchAgent -> WriterAgent",
        verbose=True,
    )

    agent_rearrange.flow = "ResearchAgent WriterAgent"

    with pytest.raises(ValueError, match="'->"):
        agent_rearrange.validate_flow()


# ============================================================================
# Flow Pattern Tests
# ============================================================================


def test_set_custom_flow():
    """Test setting custom flow."""
    agents = create_sample_agents()

    agent_rearrange = AgentRearrange(
        agents=agents,
        flow="ResearchAgent -> WriterAgent",
        verbose=True,
    )

    new_flow = "WriterAgent -> ResearchAgent -> ReviewerAgent"
    agent_rearrange.set_custom_flow(new_flow)
    assert agent_rearrange.flow == new_flow


def test_sequential_flow():
    """Test sequential flow execution."""
    agents = create_sample_agents()

    agent_rearrange = AgentRearrange(
        agents=agents,
        flow="ResearchAgent -> WriterAgent",
        max_loops=1,
        verbose=True,
    )

    result = agent_rearrange.run("What is 2+2?")
    assert result is not None


def test_concurrent_flow():
    """Test concurrent flow execution with comma syntax."""
    agents = create_sample_agents()

    agent_rearrange = AgentRearrange(
        agents=agents,
        flow="ResearchAgent, WriterAgent -> ReviewerAgent",
        max_loops=1,
        verbose=True,
    )

    result = agent_rearrange.run("What is 3+3?")
    assert result is not None


# ============================================================================
# Sequential Awareness Tests
# ============================================================================


def test_get_sequential_flow_structure():
    """Test getting sequential flow structure."""
    agents = create_sample_agents()

    agent_rearrange = AgentRearrange(
        agents=agents,
        flow="ResearchAgent -> WriterAgent -> ReviewerAgent",
        verbose=True,
    )

    flow_structure = agent_rearrange.get_sequential_flow_structure()
    assert flow_structure is not None
    assert isinstance(flow_structure, str)


def test_get_agent_sequential_awareness():
    """Test getting agent sequential awareness."""
    agents = create_sample_agents()

    agent_rearrange = AgentRearrange(
        agents=agents,
        flow="ResearchAgent -> WriterAgent -> ReviewerAgent",
        verbose=True,
    )

    awareness = agent_rearrange.get_agent_sequential_awareness(
        "WriterAgent"
    )
    assert awareness is not None
    assert isinstance(awareness, str)


# ============================================================================
# Execution Tests
# ============================================================================


def test_run_basic():
    """Test basic run execution."""
    agents = create_sample_agents()

    agent_rearrange = AgentRearrange(
        agents=agents,
        flow="ResearchAgent -> WriterAgent",
        max_loops=1,
        verbose=True,
    )

    result = agent_rearrange.run("Calculate 5+5")
    assert result is not None


def test_run_with_different_output_types():
    """Test run with different output types."""
    agents = create_sample_agents()

    for output_type in ["all", "final"]:
        agent_rearrange = AgentRearrange(
            agents=agents,
            flow="ResearchAgent -> WriterAgent",
            output_type=output_type,
            max_loops=1,
            verbose=True,
        )

        result = agent_rearrange.run("What is the capital of France?")
        assert result is not None


def test_callable_execution():
    """Test __call__ method."""
    agents = create_sample_agents()

    agent_rearrange = AgentRearrange(
        agents=agents,
        flow="ResearchAgent -> WriterAgent",
        max_loops=1,
        verbose=True,
    )

    result = agent_rearrange("What is 10+10?")
    assert result is not None


def test_concurrent_run():
    """Test concurrent execution."""
    agents = create_sample_agents()

    agent_rearrange = AgentRearrange(
        agents=agents,
        flow="ResearchAgent -> WriterAgent",
        max_loops=1,
        verbose=True,
    )

    tasks = ["What is 3+3?", "What is 4+4?"]
    results = agent_rearrange.concurrent_run(tasks, max_workers=2)

    assert results is not None
    assert len(results) == 2


# ============================================================================
# Serialization Tests
# ============================================================================


def test_to_dict():
    """Test serialization to dictionary."""
    agents = create_sample_agents()

    agent_rearrange = AgentRearrange(
        agents=agents,
        flow="ResearchAgent -> WriterAgent",
        verbose=True,
    )

    result_dict = agent_rearrange.to_dict()
    assert isinstance(result_dict, dict)
    assert "name" in result_dict
    assert "flow" in result_dict
    assert "agents" in result_dict


# ============================================================================
# Integration Tests
# ============================================================================


def test_complete_workflow():
    """Test complete workflow with all features."""
    agents = create_sample_agents()

    agent_rearrange = AgentRearrange(
        name="test-workflow",
        description="Test complete workflow",
        agents=agents,
        flow="ResearchAgent -> WriterAgent -> ReviewerAgent",
        max_loops=1,
        team_awareness=True,
        verbose=True,
        output_type="all",
    )

    result1 = agent_rearrange.run("What is Python?")
    assert result1 is not None

    flow_structure = agent_rearrange.get_sequential_flow_structure()
    assert flow_structure is not None

    result_dict = agent_rearrange.to_dict()
    assert isinstance(result_dict, dict)


# ============================================================================
# Error Handling Tests
# ============================================================================


def _make_rearrange(agents, flow, **kwargs):
    return AgentRearrange(
        agents=agents, flow=flow, max_loops=1, **kwargs
    )


def test_missing_agent_raises():
    """run() must raise when flow references a removed agent."""
    agents = create_sample_agents()
    r = _make_rearrange(agents, "ResearchAgent -> WriterAgent")
    del r.agents["WriterAgent"]

    with pytest.raises(ValueError, match="not registered"):
        r.run("test")


def test_broken_conversation_raises():
    """run() must raise when conversation is corrupted."""
    agents = create_sample_agents()
    r = _make_rearrange(agents, "ResearchAgent -> WriterAgent")
    r.conversation.conversation_history = None

    with pytest.raises((TypeError, AttributeError)):
        r.run("test")


def test_agent_error_raises():
    """run() must raise when an agent's run() raises unexpectedly."""
    agents = create_sample_agents()
    r = _make_rearrange(agents, "ResearchAgent -> WriterAgent")

    def bad_run(*args, **kwargs):
        raise TypeError("unexpected error in agent")

    r.agents["WriterAgent"].run = bad_run

    with pytest.raises(TypeError, match="unexpected error in agent"):
        r.run("test")


def test_callable_propagates():
    """__call__ must raise, not return the exception object."""
    agents = create_sample_agents()
    r = _make_rearrange(agents, "ResearchAgent -> WriterAgent")
    del r.agents["WriterAgent"]

    with pytest.raises(ValueError, match="not registered"):
        r("test")


def test_batch_run_propagates():
    """batch_run must raise, not return None."""
    agents = create_sample_agents()
    r = _make_rearrange(agents, "ResearchAgent -> WriterAgent")
    del r.agents["WriterAgent"]

    with pytest.raises(ValueError, match="not registered"):
        r.batch_run(["test1", "test2"])


def test_error_logged_once():
    """_catch_error should fire exactly once per failure."""
    agents = create_sample_agents()
    r = _make_rearrange(agents, "ResearchAgent -> WriterAgent")
    del r.agents["WriterAgent"]

    call_count = 0
    original_catch = r._catch_error

    def counting_catch(e):
        nonlocal call_count
        call_count += 1
        original_catch(e)

    r._catch_error = counting_catch

    with pytest.raises(ValueError):
        r.run("test")

    assert (
        call_count == 1
    ), f"_catch_error called {call_count} times, expected 1"


def test_successful_run_returns_result():
    """A successful run must return a non-None result."""
    agents = create_sample_agents()
    r = _make_rearrange(agents, "ResearchAgent -> WriterAgent")

    result = r.run("What is 2+2?")
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0


def test_successful_callable_returns_result():
    """__call__ on success must return a result."""
    agents = create_sample_agents()
    r = _make_rearrange(agents, "ResearchAgent -> WriterAgent")

    result = r("What is 2+2?")
    assert result is not None


# ============================================================================
# Repeated Agent Flow Tests
# ============================================================================


def create_repeated_flow_agents():
    """Create agents for repeated flow testing."""
    return [
        Agent(
            agent_name="Writer",
            agent_description="Expert in writing content",
            system_prompt="You are a writer. Write one concise sentence.",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=False,
        ),
        Agent(
            agent_name="Reviewer",
            agent_description="Expert in reviewing content",
            system_prompt="You are a reviewer. Give one critique.",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=False,
        ),
    ]


def test_repeated_agent_flow_valid():
    """Test that flows with repeated agents pass validation."""
    agents = create_repeated_flow_agents()

    agent_rearrange = AgentRearrange(
        agents=agents,
        flow="Writer -> Reviewer -> Writer",
    )

    assert agent_rearrange.validate_flow() is True


def test_repeated_agent_awareness_position_0():
    """Test that the first occurrence of a repeated agent gets correct awareness."""
    agents = create_repeated_flow_agents()

    agent_rearrange = AgentRearrange(
        agents=agents,
        flow="Writer -> Reviewer -> Writer",
    )

    tasks = agent_rearrange.flow.split("->")
    awareness = agent_rearrange._get_sequential_awareness(
        "Writer", tasks, task_idx=0
    )

    assert "Agent behind" in awareness
    assert "Reviewer" in awareness
    assert "Agent ahead" not in awareness


def test_repeated_agent_awareness_position_2():
    """Test that the second occurrence of a repeated agent gets correct awareness."""
    agents = create_repeated_flow_agents()

    agent_rearrange = AgentRearrange(
        agents=agents,
        flow="Writer -> Reviewer -> Writer",
    )

    tasks = agent_rearrange.flow.split("->")
    awareness = agent_rearrange._get_sequential_awareness(
        "Writer", tasks, task_idx=2
    )

    assert "Agent ahead" in awareness
    assert "Reviewer" in awareness
    assert "Agent behind" not in awareness


def test_repeated_agent_awareness_differs_per_position():
    """Test that each occurrence of a repeated agent gets different awareness."""
    agents = create_repeated_flow_agents()

    agent_rearrange = AgentRearrange(
        agents=agents,
        flow="Writer -> Reviewer -> Writer",
    )

    tasks = agent_rearrange.flow.split("->")
    awareness_0 = agent_rearrange._get_sequential_awareness(
        "Writer", tasks, task_idx=0
    )
    awareness_2 = agent_rearrange._get_sequential_awareness(
        "Writer", tasks, task_idx=2
    )

    assert awareness_0 != awareness_2


def test_repeated_agent_awareness_fallback_without_idx():
    """Test that awareness still works when task_idx is not provided (backward compat)."""
    agents = create_repeated_flow_agents()

    agent_rearrange = AgentRearrange(
        agents=agents,
        flow="Writer -> Reviewer -> Writer",
    )

    tasks = agent_rearrange.flow.split("->")
    awareness = agent_rearrange._get_sequential_awareness(
        "Writer", tasks
    )

    assert awareness is not None
    assert isinstance(awareness, str)
    assert "Sequential awareness" in awareness


def test_repeated_agent_three_occurrences():
    """Test awareness correctness with three occurrences of the same agent."""
    agents = create_repeated_flow_agents()

    agent_rearrange = AgentRearrange(
        agents=agents,
        flow="Writer -> Reviewer -> Writer -> Reviewer -> Writer",
    )

    tasks = agent_rearrange.flow.split("->")

    a0 = agent_rearrange._get_sequential_awareness(
        "Writer", tasks, task_idx=0
    )
    assert "Agent behind" in a0
    assert "Agent ahead" not in a0

    a2 = agent_rearrange._get_sequential_awareness(
        "Writer", tasks, task_idx=2
    )
    assert "Agent ahead" in a2
    assert "Agent behind" in a2

    a4 = agent_rearrange._get_sequential_awareness(
        "Writer", tasks, task_idx=4
    )
    assert "Agent ahead" in a4
    assert "Agent behind" not in a4


def test_repeated_agent_run():
    """Test that a repeated agent flow runs end-to-end."""
    agents = create_repeated_flow_agents()

    agent_rearrange = AgentRearrange(
        name="repeated-flow-test",
        agents=agents,
        flow="Writer -> Reviewer -> Writer",
        max_loops=1,
    )

    result = agent_rearrange.run("Write about the moon.")
    assert result is not None
    assert len(str(result)) > 0

    messages = agent_rearrange.conversation.to_dict()
    writer_msgs = [m for m in messages if m.get("role") == "Writer"]
    assert (
        len(writer_msgs) == 2
    ), f"Expected 2 Writer messages, got {len(writer_msgs)}"


def test_repeated_agent_awareness_in_conversation():
    """Test that different awareness messages are injected for each occurrence."""
    agents = create_repeated_flow_agents()

    agent_rearrange = AgentRearrange(
        name="awareness-conv-test",
        agents=agents,
        flow="Writer -> Reviewer -> Writer",
        max_loops=1,
    )

    agent_rearrange.run("Write about rain.")

    messages = agent_rearrange.conversation.to_dict()

    writer_awareness = []
    for idx, msg in enumerate(messages):
        if "Sequential awareness" in str(msg.get("content", "")):
            if (
                idx + 1 < len(messages)
                and messages[idx + 1].get("role") == "Writer"
            ):
                writer_awareness.append(msg.get("content", ""))

    assert (
        len(writer_awareness) == 2
    ), f"Expected 2 awareness messages before Writer, got {len(writer_awareness)}"
    assert (
        writer_awareness[0] != writer_awareness[1]
    ), "Both Writer invocations got identical awareness"


# ============================================================================
# batch_run — return-value correctness (mock-based)
# ============================================================================


class TestBatchRunReturns:
    def test_returns_list_same_length_as_tasks(self):
        pipeline = _make_pipeline("AgentA", "AgentB")
        tasks = ["t1", "t2", "t3"]
        results = pipeline.batch_run(tasks=tasks, batch_size=10)
        assert isinstance(results, list)
        assert len(results) == len(tasks)

    def test_single_task(self):
        pipeline = _make_pipeline("AgentA", "AgentB")
        results = pipeline.batch_run(
            tasks=["only task"], batch_size=5
        )
        assert len(results) == 1

    def test_empty_task_list(self):
        pipeline = _make_pipeline("AgentA", "AgentB")
        results = pipeline.batch_run(tasks=[], batch_size=5)
        assert results == []

    def test_results_in_input_order(self):
        """Results must match the task order, not thread-completion order."""
        pipeline = _make_pipeline("AgentA", "AgentB", delay=0.02)
        tasks = [f"task-{i}" for i in range(8)]
        results = pipeline.batch_run(tasks=tasks, batch_size=8)
        for i, result in enumerate(results):
            assert (
                f"task-{i}" in result
            ), f"result[{i}] = {result!r} does not contain task-{i}"

    def test_results_across_multiple_batches_ordered(self):
        """Order must be preserved even when tasks span multiple batches."""
        pipeline = _make_pipeline("AgentA", "AgentB")
        tasks = [f"item-{i}" for i in range(7)]
        results = pipeline.batch_run(tasks=tasks, batch_size=3)
        assert len(results) == 7
        for i, result in enumerate(results):
            assert f"item-{i}" in result


# ============================================================================
# batch_run — concurrency
# ============================================================================


class TestBatchRunConcurrency:
    def test_tasks_run_concurrently(self):
        """
        Verify concurrency by asserting that at least two tasks overlap in
        execution, rather than relying on wall-clock timing thresholds.
        """
        N = 5
        pipeline = _make_pipeline("SlowAgent", "SlowAgent2")
        tasks = [f"task-{i}" for i in range(N)]

        active_count = 0
        overlap_detected = threading.Event()
        state_lock = threading.Lock()
        original_run = pipeline.run

        def instrumented_run(task, img=None, *a, **kw):
            nonlocal active_count
            with state_lock:
                active_count += 1
                if active_count >= 2:
                    overlap_detected.set()
            try:
                time.sleep(0.05)
                return original_run(task, img, *a, **kw)
            finally:
                with state_lock:
                    active_count -= 1

        pipeline.run = instrumented_run
        results = pipeline.batch_run(tasks=tasks, batch_size=N)

        assert len(results) == N
        assert (
            overlap_detected.is_set()
        ), "Expected at least two batch_run tasks to overlap in execution"

    def test_threadpoolexecutor_is_used(self):
        """Patch ThreadPoolExecutor to confirm it is invoked for each batch."""
        pipeline = _make_pipeline("AgentA", "AgentB")
        tasks = ["t1", "t2", "t3"]

        with patch(
            "swarms.structs.agent_rearrange.ThreadPoolExecutor",
            wraps=__import__(
                "concurrent.futures", fromlist=["ThreadPoolExecutor"]
            ).ThreadPoolExecutor,
        ) as mock_tpe:
            pipeline.batch_run(tasks=tasks, batch_size=10)
            assert mock_tpe.call_count == 1

    def test_multiple_batches_uses_executor_per_batch(self):
        """One ThreadPoolExecutor context-manager per batch."""
        pipeline = _make_pipeline("AgentA", "AgentB")
        tasks = [f"t{i}" for i in range(6)]

        with patch(
            "swarms.structs.agent_rearrange.ThreadPoolExecutor",
            wraps=__import__(
                "concurrent.futures", fromlist=["ThreadPoolExecutor"]
            ).ThreadPoolExecutor,
        ) as mock_tpe:
            pipeline.batch_run(tasks=tasks, batch_size=2)
            # 6 tasks / batch_size=2 -> 3 batches -> 3 executor instances
            assert mock_tpe.call_count == 3


# ============================================================================
# batch_run — conversation isolation
# ============================================================================


class TestConversationIsolation:
    def test_original_conversation_not_mutated(self):
        """The pipeline's own conversation should not change after batch_run."""
        pipeline = _make_pipeline("AgentA", "AgentB")
        original_msg_count = len(
            pipeline.conversation.conversation_history
        )

        pipeline.batch_run(tasks=["task1", "task2"], batch_size=5)

        after_msg_count = len(
            pipeline.conversation.conversation_history
        )
        assert (
            after_msg_count == original_msg_count
        ), "pipeline.conversation was mutated by batch_run"

    def test_each_task_gets_own_conversation_copy(self):
        """Each worker clone must have its own conversation, not share one."""
        pipeline = _make_pipeline("AgentA", "AgentB")
        tasks = ["alpha", "beta", "gamma"]

        seen_conversations: list = []

        def _mock_run(self_inner, task=None, img=None, *a, **kw):
            seen_conversations.append(self_inner.conversation)
            return f"result:{task}"

        with patch.object(AgentRearrange, "_run", _mock_run):
            pipeline.batch_run(tasks=tasks, batch_size=10)

        assert len(seen_conversations) == len(tasks)
        for i in range(len(seen_conversations)):
            for j in range(i + 1, len(seen_conversations)):
                assert (
                    seen_conversations[i] is not seen_conversations[j]
                ), f"Tasks {i} and {j} shared the same conversation object"

    def test_stateful_agent_state_does_not_bleed_across_tasks(self):
        """
        Regression: concurrent tasks must not corrupt each other's results
        via shared agent state. With deepcopy each task owns its own agent
        instance, so last_task cannot be overwritten by a racing thread.
        """

        class StatefulAgent:
            def __init__(self, name: str):
                self.agent_name = name
                self.system_prompt = ""
                self.last_task = None

            def run(self, task, *args, **kwargs):
                time.sleep(0.05)
                self.last_task = task
                return f"{self.agent_name}::{self.last_task}"

        pipeline = AgentRearrange(
            agents=[StatefulAgent("A"), StatefulAgent("B")],
            flow="A -> B",
            max_loops=1,
            autosave=False,
            output_type="final",
        )

        N = 10
        tasks = [f"t{i}" for i in range(N)]
        results = pipeline.batch_run(tasks=tasks, batch_size=N)

        assert len(results) == N
        for i, result in enumerate(results):
            assert f"t{i}" in str(
                result
            ), f"result[{i}] missing 't{i}': {result!r} — agent state race detected"


# ============================================================================
# batch_run — image forwarding
# ============================================================================


class TestBatchRunWithImages:
    def test_img_list_passed_per_task(self):
        """When img is provided, each task receives its corresponding image path."""
        received: List[Optional[str]] = []

        agent_a = _make_agent("AgentA")
        agent_b = _make_agent("AgentB")

        pipeline = AgentRearrange(
            agents=[agent_a, agent_b],
            flow="AgentA -> AgentB",
            max_loops=1,
            autosave=False,
            output_type="final",
        )

        tasks = ["t1", "t2", "t3"]
        images = ["img1.png", "img2.png", "img3.png"]

        def _mock_run(self_inner, task=None, img=None, *a, **kw):
            received.append(img)
            return f"result:{task}"

        with patch.object(AgentRearrange, "_run", _mock_run):
            results = pipeline.batch_run(
                tasks=tasks, img=images, batch_size=5
            )

        assert len(results) == 3
        assert sorted(received) == sorted(images)

    def test_no_img_passes_none(self):
        """When img is omitted, None is passed as img for every task."""
        received_imgs: List[Optional[str]] = []

        agent_a = _make_agent("AgentA")
        agent_b = _make_agent("AgentB")

        pipeline = AgentRearrange(
            agents=[agent_a, agent_b],
            flow="AgentA -> AgentB",
            max_loops=1,
            autosave=False,
            output_type="final",
        )

        def _mock_run(self_inner, task=None, img=None, *a, **kw):
            received_imgs.append(img)
            return f"result:{task}"

        with patch.object(AgentRearrange, "_run", _mock_run):
            pipeline.batch_run(tasks=["t1", "t2"], batch_size=5)

        assert all(img is None for img in received_imgs)


# ============================================================================
# batch_run — batch_size validation
# ============================================================================


class TestBatchSizeBoundaries:
    @pytest.mark.parametrize("batch_size", [1, 2, 3, 5, 100])
    def test_various_batch_sizes_return_all_results(self, batch_size):
        pipeline = _make_pipeline("AgentA", "AgentB")
        tasks = [f"task-{i}" for i in range(5)]
        results = pipeline.batch_run(
            tasks=tasks, batch_size=batch_size
        )
        assert len(results) == len(tasks)

    def test_batch_size_one_still_uses_executor(self):
        """Even batch_size=1 should go through ThreadPoolExecutor."""
        pipeline = _make_pipeline("AgentA", "AgentB")
        with patch(
            "swarms.structs.agent_rearrange.ThreadPoolExecutor",
            wraps=__import__(
                "concurrent.futures", fromlist=["ThreadPoolExecutor"]
            ).ThreadPoolExecutor,
        ) as mock_tpe:
            pipeline.batch_run(tasks=["only"], batch_size=1)
            assert mock_tpe.call_count == 1

    @pytest.mark.parametrize("batch_size", [0, -1, -10])
    def test_invalid_batch_size_raises(self, batch_size):
        """batch_size <= 0 must raise ValueError immediately."""
        pipeline = _make_pipeline("AgentA", "AgentB")
        with pytest.raises(ValueError, match="batch_size"):
            pipeline.batch_run(tasks=["t1"], batch_size=batch_size)

    def test_img_length_mismatch_raises(self):
        """Mismatched img and tasks lengths must raise ValueError."""
        pipeline = _make_pipeline("AgentA", "AgentB")
        with pytest.raises(ValueError, match="img length"):
            pipeline.batch_run(
                tasks=["t1", "t2"], img=["img1.png"], batch_size=5
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
