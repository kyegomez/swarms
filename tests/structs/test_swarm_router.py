import pytest

from swarms.structs.swarm_router import (
    SwarmRouter,
    SwarmRouterConfig,
    SwarmRouterRunError,
    SwarmRouterConfigError,
    Document,
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
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=False,
            print_on=False,
        ),
        Agent(
            agent_name="CodeAgent",
            agent_description="Expert in coding",
            system_prompt="You are a coding expert.",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=False,
            print_on=False,
        ),
    ]


# ============================================================================
# Initialization Tests
# ============================================================================


def test_default_initialization():
    """Test SwarmRouter with default parameters."""
    router = SwarmRouter()

    assert router.name == "swarm-router"
    assert (
        router.description == "Routes your task to the desired swarm"
    )
    assert router.max_loops == 1
    assert router.agents == []
    assert router.swarm_type == "SequentialWorkflow"
    assert router.autosave is False
    assert router.return_json is False
    assert router.auto_generate_prompts is False
    assert router.shared_memory_system is None
    assert router.rules is None
    assert router.documents == []
    assert router.output_type == "dict-all-except-first"
    assert router.verbose is False
    assert router.telemetry_enabled is False


def test_custom_initialization():
    """Test SwarmRouter with custom parameters."""
    sample_agents = create_sample_agents()

    router = SwarmRouter(
        name="test-router",
        description="Test router description",
        max_loops=3,
        agents=sample_agents,
        swarm_type="ConcurrentWorkflow",
        autosave=True,
        return_json=True,
        auto_generate_prompts=True,
        rules="Test rules",
        documents=["doc1.txt", "doc2.txt"],
        output_type="json",
        verbose=False,  # Keep quiet for tests
        telemetry_enabled=False,
    )

    assert router.name == "test-router"
    assert router.description == "Test router description"
    assert router.max_loops == 3
    assert router.agents == sample_agents
    assert router.swarm_type == "ConcurrentWorkflow"
    assert router.autosave is True
    assert router.return_json is True
    assert router.auto_generate_prompts is True
    assert router.rules == "Test rules"
    assert router.documents == ["doc1.txt", "doc2.txt"]
    assert router.output_type == "json"
    assert router.verbose is False
    assert router.telemetry_enabled is False


def test_initialization_with_heavy_swarm_config():
    """Test SwarmRouter with HeavySwarm specific configuration."""
    sample_agents = create_sample_agents()

    router = SwarmRouter(
        agents=sample_agents,
        swarm_type="HeavySwarm",
        heavy_swarm_loops_per_agent=2,
        heavy_swarm_question_agent_model_name="gpt-4o-mini",
        heavy_swarm_worker_model_name="gpt-4o-mini",
        heavy_swarm_swarm_show_output=False,
    )

    assert router.swarm_type == "HeavySwarm"
    assert router.heavy_swarm_loops_per_agent == 2
    assert (
        router.heavy_swarm_question_agent_model_name == "gpt-4o-mini"
    )
    assert router.heavy_swarm_worker_model_name == "gpt-4o-mini"
    assert router.heavy_swarm_swarm_show_output is False


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


def test_initialization_with_shared_memory():
    """Test SwarmRouter with shared memory system."""
    sample_agents = create_sample_agents()

    router = SwarmRouter(
        agents=sample_agents,
        shared_memory_system=None,  # Test with None for now
    )

    assert router.shared_memory_system is None


def test_initialization_with_worker_tools():
    """Test SwarmRouter with worker tools."""
    sample_agents = create_sample_agents()

    router = SwarmRouter(
        agents=sample_agents,
        worker_tools=[],  # Empty list for now
    )

    assert router.worker_tools == []


# ============================================================================
# Document Management Tests
# ============================================================================


def test_document_creation():
    """Test Document creation."""
    doc = Document(
        file_path="/path/to/test/document.txt",
        data="This is test content",
    )

    assert doc.file_path == "/path/to/test/document.txt"
    assert doc.data == "This is test content"


def test_router_with_documents():
    """Test SwarmRouter with document configuration."""
    sample_agents = create_sample_agents()
    documents = [
        Document(file_path="/path/to/doc1.txt", data="Content1"),
        Document(file_path="/path/to/doc2.txt", data="Content2"),
    ]

    router = SwarmRouter(
        agents=sample_agents,
        documents=documents,
    )

    assert len(router.documents) == 2
    assert router.documents[0].file_path == "/path/to/doc1.txt"
    assert router.documents[1].file_path == "/path/to/doc2.txt"


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
        rules=None,
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
        rules="Test rules",
        multi_agent_collab_prompt=False,
        task="Test task",
    )

    # SwarmRouter doesn't accept config directly, but we can verify config is valid
    assert config.name == "config-router"
    assert config.description == "Router from config"
    assert config.swarm_type == "SequentialWorkflow"
    assert config.rules == "Test rules"

    # Create router with matching parameters
    router = SwarmRouter(
        name=config.name,
        description=config.description,
        agents=sample_agents,
        swarm_type=config.swarm_type,
        rules=config.rules,
    )

    assert router.name == config.name
    assert router.description == config.description
    assert router.swarm_type == config.swarm_type
    assert router.rules == config.rules


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


def test_invalid_swarm_type():
    """Test router with invalid swarm type."""
    sample_agents = create_sample_agents()

    # This should not raise an error during initialization
    router = SwarmRouter(
        agents=sample_agents,
        swarm_type="InvalidSwarmType",
    )

    # But should raise ValueError during execution when creating swarm
    with pytest.raises(
        ValueError, match="Invalid swarm type: InvalidSwarmType"
    ):
        router.run("Test task")


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
