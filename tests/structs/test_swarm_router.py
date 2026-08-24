import os
import pytest

from swarms.structs.swarm_router import (
    SwarmRouter,
    SwarmRouterConfig,
    SwarmRouterRunError,
    SwarmRouterConfigError,
)
from swarms.structs.agent import Agent

# ============================================================================
# Helper Functions
# ============================================================================


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



# Live-LLM tests: they call agent.run() against a real model and can only
# pass with an API key. Without one, Agent.run now honestly raises
# AgentLLMError after retry exhaustion, so these tests are skipped instead
# of failing (same convention as tests/telemetry/test_telemetry.py).
_LLM_KEYS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GROQ_API_KEY",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
)
_HAS_LLM_KEY = any(os.getenv(k) for k in _LLM_KEYS)
requires_llm = pytest.mark.skipif(
    not _HAS_LLM_KEY,
    reason="no LLM API key set (OPENAI_API_KEY etc.) - live-LLM test skipped",
)

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


@requires_llm
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


@requires_llm
def test_run_with_empty_task():
    """Test running SwarmRouter with empty task."""
    sample_agents = create_sample_agents()

    router = SwarmRouter(agents=sample_agents, verbose=False)

    # Empty task is allowed, router will pass it to the swarm
    result = router.run("")
    assert result is not None


@requires_llm
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


@requires_llm
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


@requires_llm
def test_call_method():
    """Test __call__ method."""
    sample_agents = create_sample_agents()

    router = SwarmRouter(
        agents=sample_agents,
        verbose=False,
    )

    result = router("What is the capital of France?")
    assert result is not None


@requires_llm
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


@requires_llm
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


@requires_llm
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


@requires_llm
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


@requires_llm
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


@requires_llm
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


def test_run_with_auto():
    """SwarmRouter dispatches to 'auto' (embedding-based selection)."""
    sample_agents = create_sample_agents()

    router = SwarmRouter(
        agents=sample_agents,
        swarm_type="auto",
        max_loops=1,
        verbose=False,
    )

    result = router.run("What is 1+1?")
    assert result is not None


@requires_llm
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


@requires_llm
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


def test_run_with_batched_grid_workflow():
    """SwarmRouter dispatches to BatchedGridWorkflow."""
    sample_agents = create_sample_agents()

    router = SwarmRouter(
        agents=sample_agents,
        swarm_type="BatchedGridWorkflow",
        max_loops=1,
        verbose=False,
    )

    result = router.run("What is 1+1?")
    assert result is not None


@requires_llm
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
