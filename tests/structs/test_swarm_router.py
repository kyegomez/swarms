from unittest.mock import patch

import pytest
from typing import get_args

from swarms.structs.swarm_router import (
    SwarmRouter,
    SwarmRouterConfig,
    SwarmRouterRunError,
    SwarmRouterConfigError,
    SwarmType,
)
from swarms.structs.agent import Agent


def create_sample_agents():
    """Create sample agents for testing."""
    return [
        Agent(
            agent_name="ResearchAgent",
            agent_description="Specializes in researching topics",
            system_prompt="You are a research specialist.",
            model_name="gpt-5.4",
            max_loops=1,
            verbose=False,
            print_on=False,
        ),
        Agent(
            agent_name="CodeAgent",
            agent_description="Expert in coding",
            system_prompt="You are a coding expert.",
            model_name="gpt-5.4",
            max_loops=1,
            verbose=False,
            print_on=False,
        ),
    ]


# ============================================================================
# Initialization Tests
# ============================================================================


def test_initialization_with_heavy_swarm_config():
    """Test SwarmRouter with HeavySwarm specific configuration."""
    sample_agents = create_sample_agents()

    router = SwarmRouter(
        agents=sample_agents,
        swarm_type="HeavySwarm",
        heavy_swarm_max_loops=2,
        heavy_swarm_question_agent_model_name="gpt-5.4",
        heavy_swarm_worker_model_name="gpt-5.4",
        heavy_swarm_swarm_show_output=False,
        heavy_swarm_variant="heavy",
    )

    assert router.swarm_type == "HeavySwarm"
    assert router.heavy_swarm_max_loops == 2
    assert router.heavy_swarm_question_agent_model_name == "gpt-5.4"
    assert router.heavy_swarm_worker_model_name == "gpt-5.4"
    assert router.heavy_swarm_swarm_show_output is False
    assert router.heavy_swarm_variant == "heavy"


def test_initialization_with_agent_rearrange_config():
    """Test SwarmRouter with AgentRearrange specific configuration."""
    sample_agents = create_sample_agents()

    router = SwarmRouter(
        agents=sample_agents,
        swarm_type="AgentRearrange",
        rearrange_flow="ResearchAgent -> CodeAgent",
    )

    assert router.swarm_type == "AgentRearrange"
    assert router.rearrange_flow == "ResearchAgent -> CodeAgent"


# ============================================================================
# Configuration Tests
# ============================================================================


def test_initialization_with_worker_tools():
    """Test SwarmRouter with worker tools."""
    sample_agents = create_sample_agents()

    router = SwarmRouter(
        agents=sample_agents,
        worker_tools=[],  # Empty list for now
    )

    assert router.worker_tools == []


# ============================================================================
# Configuration Class Tests
# ============================================================================


def test_swarm_router_config_creation():
    """Test SwarmRouterConfig creation."""
    config = SwarmRouterConfig(
        name="test-config",
        description="Test configuration",
        swarm_type="SequentialWorkflow",
        rearrange_flow=None,
        multi_agent_collab_prompt=True,
        task="Test task",
    )

    assert config.name == "test-config"
    assert config.description == "Test configuration"
    assert config.swarm_type == "SequentialWorkflow"
    assert config.task == "Test task"


def test_router_with_config():
    """Test SwarmRouter initialization matches config structure."""
    sample_agents = create_sample_agents()
    config = SwarmRouterConfig(
        name="config-router",
        description="Router from config",
        swarm_type="SequentialWorkflow",
        rearrange_flow=None,
        multi_agent_collab_prompt=False,
        task="Test task",
    )

    # SwarmRouter doesn't accept config directly, but we can verify config is valid
    assert config.name == "config-router"
    assert config.description == "Router from config"
    assert config.swarm_type == "SequentialWorkflow"

    # Create router with matching parameters
    router = SwarmRouter(
        name=config.name,
        description=config.description,
        agents=sample_agents,
        swarm_type=config.swarm_type,
    )

    assert router.name == config.name
    assert router.description == config.description
    assert router.swarm_type == config.swarm_type


# ============================================================================
# Basic Execution Tests
# ============================================================================


def test_run_with_sequential_workflow():
    """Test running SwarmRouter with SequentialWorkflow."""
    sample_agents = create_sample_agents()

    router = SwarmRouter(
        agents=sample_agents,
        swarm_type="SequentialWorkflow",
        verbose=False,
    )

    result = router.run("What is 2+2?")
    assert result is not None


def test_run_with_no_agents():
    """Test running SwarmRouter with no agents."""
    router = SwarmRouter()

    with pytest.raises(RuntimeError):
        router.run("Test task")


def test_run_with_empty_task():
    """Test running SwarmRouter with empty task."""
    sample_agents = create_sample_agents()

    router = SwarmRouter(agents=sample_agents, verbose=False)

    # Empty task is allowed, router will pass it to the swarm
    result = router.run("")
    assert result is not None


def test_run_with_none_task():
    """Test running SwarmRouter with None task."""
    sample_agents = create_sample_agents()

    router = SwarmRouter(agents=sample_agents, verbose=False)

    # None task is allowed, router will pass it to the swarm
    result = router.run(None)
    assert result is not None


# ============================================================================
# Batch Processing Tests
# ============================================================================


def test_batch_run_with_tasks():
    """Test batch processing with multiple tasks."""
    sample_agents = create_sample_agents()

    router = SwarmRouter(
        agents=sample_agents,
        verbose=False,
    )

    tasks = ["What is 1+1?", "What is 2+2?"]
    results = router.batch_run(tasks)

    assert len(results) == 2
    assert all(result is not None for result in results)


def test_batch_run_with_empty_tasks():
    """Test batch processing with empty task list."""
    sample_agents = create_sample_agents()

    router = SwarmRouter(agents=sample_agents)

    results = router.batch_run([])
    assert results == []


def test_batch_run_with_no_agents():
    """Test batch processing with no agents."""
    router = SwarmRouter()

    with pytest.raises(RuntimeError):
        router.batch_run(["Test task"])


# ============================================================================
# Call Method Tests
# ============================================================================


def test_call_method():
    """Test __call__ method."""
    sample_agents = create_sample_agents()

    router = SwarmRouter(
        agents=sample_agents,
        verbose=False,
    )

    result = router("What is the capital of France?")
    assert result is not None


def test_call_with_image():
    """Test __call__ method with image."""
    sample_agents = create_sample_agents()

    router = SwarmRouter(
        agents=sample_agents,
        verbose=False,
    )

    # Test with None image (no actual image processing)
    result = router("Describe this image", img=None)
    assert result is not None


# ============================================================================
# Output Type Tests
# ============================================================================


def test_different_output_types():
    """Test router with different output types."""
    sample_agents = create_sample_agents()

    for output_type in ["dict", "json", "string", "list"]:
        router = SwarmRouter(
            agents=sample_agents,
            output_type=output_type,
            verbose=False,
        )

        result = router.run("Simple test task")
        assert result is not None


# ============================================================================
# Error Handling Tests
# ============================================================================


def test_swarm_router_run_error():
    """Test SwarmRouterRunError exception."""
    error = SwarmRouterRunError("Test error message")
    assert str(error) == "Test error message"


def test_swarm_router_config_error():
    """Test SwarmRouterConfigError exception."""
    error = SwarmRouterConfigError("Config error message")
    assert str(error) == "Config error message"


# ============================================================================
# Integration Tests
# ============================================================================


def test_complete_workflow():
    """Test complete workflow from initialization to execution."""
    # Create agents
    agents = create_sample_agents()

    # Create router with configuration
    router = SwarmRouter(
        name="integration-test-router",
        description="Router for integration testing",
        agents=agents,
        swarm_type="SequentialWorkflow",
        max_loops=1,
        verbose=False,
        output_type="string",
    )

    # Execute single task
    result = router.run("Calculate the sum of 5 and 7")
    assert result is not None

    # Execute batch tasks
    tasks = [
        "What is 10 + 15?",
        "What is 20 - 8?",
        "What is 6 * 7?",
    ]
    batch_results = router.batch_run(tasks)

    assert len(batch_results) == 3
    assert all(result is not None for result in batch_results)


def test_router_reconfiguration():
    """Test reconfiguring router after initialization."""
    sample_agents = create_sample_agents()

    router = SwarmRouter(agents=sample_agents)

    # Change configuration
    router.max_loops = 3
    router.output_type = "json"
    router.verbose = False

    assert router.max_loops == 3
    assert router.output_type == "json"
    assert router.verbose is False

    # Test execution with new configuration
    result = router.run("Test reconfiguration")
    assert result is not None


# ============================================================================
# Swarm Type Coverage — one .run() per supported swarm_type
# ============================================================================
#
# These tests exercise the SwarmRouter dispatch end-to-end for every type the
# router claims to support. Each test uses minimal config and a trivial task
# to keep LLM cost down; we only assert that .run() returns something, since
# correctness of each underlying swarm is its own test file's responsibility.


def test_run_with_agent_rearrange():
    """SwarmRouter dispatches to AgentRearrange."""
    sample_agents = create_sample_agents()

    router = SwarmRouter(
        agents=sample_agents,
        swarm_type="AgentRearrange",
        rearrange_flow="ResearchAgent -> CodeAgent",
        max_loops=1,
        verbose=False,
    )

    result = router.run("What is 1+1?")
    assert result is not None


def test_run_with_mixture_of_agents():
    """SwarmRouter dispatches to MixtureOfAgents."""
    sample_agents = create_sample_agents()

    router = SwarmRouter(
        agents=sample_agents,
        swarm_type="MixtureOfAgents",
        max_loops=1,
        verbose=False,
    )

    result = router.run("What is 1+1?")
    assert result is not None


def test_run_with_sequential_workflow_type():
    """SwarmRouter dispatches to SequentialWorkflow."""
    sample_agents = create_sample_agents()

    router = SwarmRouter(
        agents=sample_agents,
        swarm_type="SequentialWorkflow",
        max_loops=1,
        verbose=False,
    )

    result = router.run("What is 1+1?")
    assert result is not None


def test_run_with_concurrent_workflow():
    """SwarmRouter dispatches to ConcurrentWorkflow."""
    sample_agents = create_sample_agents()

    router = SwarmRouter(
        agents=sample_agents,
        swarm_type="ConcurrentWorkflow",
        max_loops=1,
        verbose=False,
    )

    result = router.run("What is 1+1?")
    assert result is not None


def test_run_with_group_chat():
    """SwarmRouter dispatches to GroupChat."""
    sample_agents = create_sample_agents()

    router = SwarmRouter(
        agents=sample_agents,
        swarm_type="GroupChat",
        max_loops=1,
        verbose=False,
    )

    result = router.run("What is 1+1?")
    assert result is not None


def test_run_with_multi_agent_router():
    """SwarmRouter dispatches to MultiAgentRouter."""
    sample_agents = create_sample_agents()

    router = SwarmRouter(
        agents=sample_agents,
        swarm_type="MultiAgentRouter",
        max_loops=1,
        verbose=False,
    )

    result = router.run("What is 1+1?")
    assert result is not None


def test_run_with_hierarchical_swarm():
    """SwarmRouter dispatches to HierarchicalSwarm."""
    sample_agents = create_sample_agents()

    router = SwarmRouter(
        agents=sample_agents,
        swarm_type="HierarchicalSwarm",
        max_loops=1,
        verbose=False,
    )

    result = router.run("What is 1+1?")
    assert result is not None


def test_auto_is_rejected_at_construction():
    from swarms.structs.swarm_router import (
        SwarmRouterConfigError,
        SwarmType,
    )

    assert "auto" not in get_args(SwarmType)

    with pytest.raises(SwarmRouterConfigError):
        SwarmRouter(
            agents=create_sample_agents(),
            swarm_type="auto",
            max_loops=1,
            verbose=False,
        )


def test_run_with_majority_voting():
    """SwarmRouter dispatches to MajorityVoting."""
    sample_agents = create_sample_agents()

    router = SwarmRouter(
        agents=sample_agents,
        swarm_type="MajorityVoting",
        max_loops=1,
        verbose=False,
    )

    result = router.run("What is 1+1?")
    assert result is not None


def test_run_with_council_as_judge():
    """SwarmRouter dispatches to CouncilAsAJudge."""
    sample_agents = create_sample_agents()

    router = SwarmRouter(
        agents=sample_agents,
        swarm_type="CouncilAsAJudge",
        max_loops=1,
        verbose=False,
    )

    result = router.run("What is 1+1?")
    assert result is not None


def test_run_with_heavy_swarm():
    """SwarmRouter dispatches to HeavySwarm."""
    sample_agents = create_sample_agents()

    router = SwarmRouter(
        agents=sample_agents,
        swarm_type="HeavySwarm",
        heavy_swarm_max_loops=1,
        heavy_swarm_swarm_show_output=False,
        max_loops=1,
        verbose=False,
    )

    result = router.run("What is 1+1?")
    assert result is not None


def test_batched_grid_workflow_is_rejected_at_construction():
    """BatchedGridWorkflow is not routable and is not offered as a SwarmType."""
    assert "BatchedGridWorkflow" not in get_args(SwarmType)

    with pytest.raises(SwarmRouterConfigError):
        SwarmRouter(
            agents=create_sample_agents(),
            swarm_type="BatchedGridWorkflow",
            max_loops=1,
            verbose=False,
        )


def test_run_with_llm_council():
    """SwarmRouter dispatches to LLMCouncil."""
    sample_agents = create_sample_agents()

    router = SwarmRouter(
        agents=sample_agents,
        swarm_type="LLMCouncil",
        max_loops=1,
        verbose=False,
    )

    result = router.run("What is 1+1?")
    assert result is not None


class TestConcurrentRun:
    """``concurrent_run`` runs a task list in parallel, in task order."""

    @staticmethod
    def _router(run_impl):
        from unittest.mock import patch

        from swarms import Agent, SwarmRouter

        with patch("swarms.structs.agent.LiteLLM"):
            agent = Agent(
                agent_name="A",
                model_name="gpt-5.4",
                max_loops=1,
                autosave=False,
                print_on=False,
            )
            router = SwarmRouter(
                name="r",
                agents=[agent],
                swarm_type="SequentialWorkflow",
                autosave=False,
            )
        router.run = run_impl
        return router

    def test_returns_one_result_per_task(self):
        r = self._router(lambda task=None, **kw: f"ran:{task}")
        assert r.concurrent_run(["a", "b"]) == ["ran:a", "ran:b"]

    def test_results_are_in_task_order(self):
        """The slowest task is first; it must still come back first."""
        import time

        def slow_first(task=None, **kw):
            time.sleep(0.05 if task == "0" else 0)
            return task

        r = self._router(slow_first)
        tasks = [str(i) for i in range(5)]
        assert r.concurrent_run(tasks) == tasks

    def test_tasks_actually_run_in_parallel(self):
        import time

        def slow(task=None, **kw):
            time.sleep(0.05)
            return task

        r = self._router(slow)
        start = time.time()
        r.concurrent_run([str(i) for i in range(6)])
        assert time.time() - start < 0.2

    def test_imgs_are_paired_with_tasks_by_position(self):
        seen = []

        def spy(task=None, img=None, **kw):
            seen.append((task, img))
            return task

        r = self._router(spy)
        r.concurrent_run(["a", "b"], imgs=["one.png", "two.png"])
        assert sorted(seen) == [
            ("a", "one.png"),
            ("b", "two.png"),
        ]

    def test_mismatched_imgs_length_raises(self):
        """Zipping would silently drop the extra task instead."""
        r = self._router(lambda task=None, **kw: task)
        with pytest.raises(ValueError, match="one image per task"):
            r.concurrent_run(["a", "b"], imgs=["only-one.png"])

    def test_empty_task_list(self):
        r = self._router(lambda task=None, **kw: task)
        assert r.concurrent_run([]) == []

    def test_exceptions_propagate(self):
        def boom(task=None, **kw):
            raise ValueError("nope")

        r = self._router(boom)
        with pytest.raises(ValueError, match="nope"):
            r.concurrent_run(["a"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ============================================================================
# multi_agent_collab_prompt — delivered, not welded onto the caller's agents
# ============================================================================


def _collab_recording_agents(names):
    """Agents whose run() records the system turns it was handed."""
    from swarms import Agent

    calls = []
    agents = []
    for name in names:
        agent = Agent(
            agent_name=name,
            system_prompt=f"You are {name}.",
            model_name="gpt-5.4",
            max_loops=1,
            verbose=False,
        )

        def _make(agent_obj, agent_name):
            def _run(task=None, messages=None, **kwargs):
                calls.append(
                    {
                        "agent": agent_name,
                        "messages": list(messages or []),
                    }
                )
                answer = f"{agent_name}-answer"
                agent_obj.short_memory.add(
                    role=agent_name, content=answer
                )
                return answer

            return _run

        agent.run = _make(agent, name)
        agents.append(agent)
    return agents, calls


def test_collab_prompt_never_mutates_the_callers_agents():
    """Building routers must not append to the caller's system_prompt.

    The preamble used to be appended with ``+=`` at construction. It never
    reached the model (the prompt is baked into the LLM when the agent is
    built) and it accumulated once per router.
    """
    agents, _ = _collab_recording_agents(["A", "B"])
    originals = [a.system_prompt for a in agents]

    SwarmRouter(
        agents=agents,
        swarm_type="SequentialWorkflow",
        multi_agent_collab_prompt=True,
    )
    SwarmRouter(
        agents=agents,
        swarm_type="SequentialWorkflow",
        multi_agent_collab_prompt=True,
    )

    assert [a.system_prompt for a in agents] == originals


def test_collab_prompt_is_delivered_as_a_system_turn():
    """The preamble must actually reach the agent at run time."""
    from swarms.prompts.multi_agent_collab_prompt import (
        MULTI_AGENT_COLLAB_PROMPT_TWO,
    )

    agents, calls = _collab_recording_agents(["A", "B"])
    router = SwarmRouter(
        agents=agents,
        swarm_type="SequentialWorkflow",
        multi_agent_collab_prompt=True,
    )
    router.run("Go.")

    delivered = [
        m["content"]
        for call in calls
        for m in call["messages"]
        if m["role"] == "system"
    ]
    assert any(
        MULTI_AGENT_COLLAB_PROMPT_TWO in text for text in delivered
    )
    assert [a.system_prompt for a in agents] == [
        "You are A.",
        "You are B.",
    ]


def test_collab_prompt_warns_when_the_swarm_type_cannot_deliver_it():
    """A flag that does nothing must say so, regardless of verbose."""
    agents, _ = _collab_recording_agents(["A", "B"])

    with patch("swarms.structs.swarm_router.logger.warning") as warn:
        SwarmRouter(
            agents=agents,
            swarm_type="ConcurrentWorkflow",
            multi_agent_collab_prompt=True,
            verbose=False,
        )

    assert warn.called
    assert "multi_agent_collab_prompt is ignored" in str(
        warn.call_args
    )


def test_list_all_agents_does_not_crash_at_construction():
    """The swarm is built lazily, so setup() must not reach for it.

    ``setup()`` used to call ``list_agents_to_eachother()``, which reads
    ``self.swarm`` — created only on the first ``run()`` — so constructing a
    router with ``list_all_agents=True`` raised ``AttributeError``.
    """
    agents, _ = _collab_recording_agents(["A", "B"])

    router = SwarmRouter(
        agents=agents,
        swarm_type="SequentialWorkflow",
        list_all_agents=True,
    )

    assert router.swarm is None


def test_list_all_agents_delivers_the_roster_as_a_system_turn():
    """The roster reaches agents, and the caller's agents are untouched.

    It cannot be seeded into the shared conversation: structures reset that
    conversation per task, which would discard it before any agent ran.
    """
    agents, calls = _collab_recording_agents(["A", "B"])
    originals = [a.system_prompt for a in agents]

    router = SwarmRouter(
        agents=agents,
        swarm_type="SequentialWorkflow",
        list_all_agents=True,
        multi_agent_collab_prompt=False,
    )
    router.run("Go.")

    delivered = [
        m["content"]
        for call in calls
        for m in call["messages"]
        if m["role"] == "system"
    ]
    assert any("Total Agents" in text for text in delivered)
    assert [a.system_prompt for a in agents] == originals


def test_conversation_points_at_the_live_swarm_conversation_after_run():
    """``router.conversation`` must be the conversation the run actually used."""
    agents, _ = _collab_recording_agents(["A", "B"])
    router = SwarmRouter(
        agents=agents, swarm_type="SequentialWorkflow", max_loops=1
    )

    assert router.conversation is None
    router.run("Go.")

    roles = [
        m["role"] for m in router.conversation.conversation_history
    ]
    assert "A" in roles and "B" in roles


# ============================================================================
# fallback_swarms
# ============================================================================


class _FakeSwarm:
    """Stands in for any swarm the factory would build."""

    def __init__(self, name, error=None):
        self.name = name
        self.error = error
        self.calls = []
        self.conversation = None
        # SequentialWorkflow exposes its conversation via agent_rearrange.
        self.agent_rearrange = self

    def run(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return f"{self.name}-result"


def _fallback_router(primary_error=None, fallbacks=None, **outcomes):
    """A router whose factory builds ``_FakeSwarm``s instead of real swarms.

    ``outcomes`` maps swarm type -> exception to raise from ``run()`` (or
    ``None`` to succeed). Types not named succeed.
    """
    with patch("swarms.structs.agent.LiteLLM"):
        agent = Agent(
            agent_name="A",
            model_name="gpt-5.4",
            max_loops=1,
            autosave=False,
            print_on=False,
        )
        router = SwarmRouter(
            name="fallback-router",
            agents=[agent],
            swarm_type="SequentialWorkflow",
            fallback_swarms=fallbacks,
            autosave=False,
        )
    outcomes.setdefault("SequentialWorkflow", primary_error)
    built = {}

    def _factory_for(swarm_type):
        def _build(*args, **kwargs):
            built[swarm_type] = _FakeSwarm(
                swarm_type, outcomes.get(swarm_type)
            )
            return built[swarm_type]

        return _build

    router._swarm_factory = {
        swarm_type: _factory_for(swarm_type)
        for swarm_type in router._swarm_factory
    }
    return router, built


def test_fallback_is_not_touched_when_the_primary_succeeds():
    router, built = _fallback_router(
        fallbacks=["ConcurrentWorkflow", "GroupChat"]
    )

    assert router.run("go") == "SequentialWorkflow-result"
    assert list(built) == ["SequentialWorkflow"]
    assert router.active_swarm_type == "SequentialWorkflow"
    assert router.fallback_attempts == []


def test_primary_failure_runs_the_first_fallback():
    boom = RuntimeError("primary down")
    router, built = _fallback_router(
        primary_error=boom,
        fallbacks=["ConcurrentWorkflow", "GroupChat"],
    )

    assert router.run("go") == "ConcurrentWorkflow-result"
    assert list(built) == ["SequentialWorkflow", "ConcurrentWorkflow"]
    assert router.active_swarm_type == "ConcurrentWorkflow"
    assert router.swarm is built["ConcurrentWorkflow"]
    assert router.fallback_attempts == [
        {"swarm_type": "SequentialWorkflow", "error": boom}
    ]


def test_fallbacks_are_tried_in_list_order():
    router, built = _fallback_router(
        primary_error=RuntimeError("1"),
        fallbacks=[
            "ConcurrentWorkflow",
            "GroupChat",
            "MajorityVoting",
        ],
        ConcurrentWorkflow=RuntimeError("2"),
        GroupChat=RuntimeError("3"),
    )

    assert router.run("go") == "MajorityVoting-result"
    assert [a["swarm_type"] for a in router.fallback_attempts] == [
        "SequentialWorkflow",
        "ConcurrentWorkflow",
        "GroupChat",
    ]
    assert "MajorityVoting" in built


def test_every_swarm_failing_raises_with_the_whole_chain():
    last = ValueError("last one")
    router, _ = _fallback_router(
        primary_error=RuntimeError("first"),
        fallbacks=["ConcurrentWorkflow"],
        ConcurrentWorkflow=last,
    )

    with pytest.raises(SwarmRouterRunError) as excinfo:
        router.run("go")

    message = str(excinfo.value)
    assert "SequentialWorkflow: RuntimeError: first" in message
    assert "ConcurrentWorkflow: ValueError: last one" in message
    assert excinfo.value.__cause__ is last
    assert len(router.fallback_attempts) == 2


def test_without_fallbacks_the_original_error_propagates_unchanged():
    boom = KeyError("unchanged")
    router, _ = _fallback_router(primary_error=boom)

    with pytest.raises(KeyError) as excinfo:
        router.run("go")

    assert excinfo.value is boom
    assert router.fallback_attempts == []


def test_a_swarm_that_fails_to_construct_also_falls_back():
    router, built = _fallback_router(fallbacks=["ConcurrentWorkflow"])

    def _broken_factory(*args, **kwargs):
        raise TypeError("cannot build")

    router._swarm_factory["SequentialWorkflow"] = _broken_factory

    assert router.run("go") == "ConcurrentWorkflow-result"
    assert router.fallback_attempts[0]["swarm_type"] == (
        "SequentialWorkflow"
    )
    assert isinstance(
        router.fallback_attempts[0]["error"], RuntimeError
    )


def test_the_run_payload_reaches_the_fallback_swarm():
    router, built = _fallback_router(
        primary_error=RuntimeError("x"),
        fallbacks=["ConcurrentWorkflow"],
    )

    router.run("go", img="chart.png")

    assert built["ConcurrentWorkflow"].calls == [
        {"task": "go", "img": "chart.png"}
    ]


def test_each_swarm_type_is_cached_under_its_own_key():
    router, built = _fallback_router(
        primary_error=RuntimeError("x"),
        fallbacks=["ConcurrentWorkflow"],
    )

    router.run("one")
    router.run("two")

    # Built once each; the second run reused both cached swarms.
    assert list(built) == ["SequentialWorkflow", "ConcurrentWorkflow"]
    assert len(built["ConcurrentWorkflow"].calls) == 2
    assert {key[0] for key in router._swarm_cache} == {
        "SequentialWorkflow",
        "ConcurrentWorkflow",
    }


def test_fallback_swarms_must_be_a_list():
    with pytest.raises(
        SwarmRouterConfigError, match="must be a list"
    ):
        SwarmRouter(
            agents=create_sample_agents(),
            swarm_type="SequentialWorkflow",
            fallback_swarms="ConcurrentWorkflow",
        )


def test_fallback_swarms_entries_must_be_valid_swarm_types():
    with pytest.raises(
        SwarmRouterConfigError, match="not a valid swarm type"
    ):
        SwarmRouter(
            agents=create_sample_agents(),
            swarm_type="SequentialWorkflow",
            fallback_swarms=["ConcurrentWorkflow", "NoSuchSwarm"],
        )


def test_agent_rearrange_fallback_requires_a_flow():
    with pytest.raises(
        SwarmRouterConfigError, match="requires rearrange_flow"
    ):
        SwarmRouter(
            agents=create_sample_agents(),
            swarm_type="SequentialWorkflow",
            fallback_swarms=["AgentRearrange"],
        )


def test_config_model_accepts_fallback_swarms():
    config = SwarmRouterConfig(
        name="r",
        description="d",
        swarm_type="SequentialWorkflow",
        rearrange_flow=None,
        multi_agent_collab_prompt=False,
        fallback_swarms=["ConcurrentWorkflow"],
        task="t",
    )
    assert config.fallback_swarms == ["ConcurrentWorkflow"]
