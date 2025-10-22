import socket
from unittest.mock import Mock, patch

import pytest

from swarms.structs.agent import Agent
from swarms.structs.aop import (
    AOP,
    AOPCluster,
    QueueStatus,
    TaskStatus,
)


@pytest.fixture
def real_agent():
    """Create a real agent for testing using example.py configuration."""
    from swarms import Agent

    agent = Agent(
        agent_name="Test-Agent",
        agent_description="Test agent for AOP testing",
        model_name="gpt-3.5-turbo",
        max_loops=1,
        temperature=0.5,
        max_tokens=4096,
    )
    return agent


@pytest.fixture
def real_agents():
    """Create multiple real agents for batch testing."""
    from swarms import Agent

    agents = []
    for i in range(3):
        agent = Agent(
            agent_name=f"Test-Agent-{i}",
            agent_description=f"Test agent {i} for AOP testing",
            model_name="gpt-3.5-turbo",
            max_loops=1,
            temperature=0.5,
            max_tokens=4096,
        )
        agents.append(agent)
    return agents


@pytest.fixture
def mock_fastmcp():
    """Create a mock FastMCP server."""
    mcp = Mock()
    mcp.name = "Test AOP"
    mcp.port = 8000
    mcp.log_level = "INFO"
    mcp.run = Mock()
    mcp.tool = Mock()
    return mcp


@pytest.fixture
def aop_instance(real_agent, mock_fastmcp):
    """Create an AOP instance for testing."""
    with patch(
        "swarms.structs.aop.FastMCP", return_value=mock_fastmcp
    ), patch("swarms.structs.aop.logger"):
        aop = AOP(
            server_name="Test AOP",
            description="Test AOP description",
            agents=[real_agent],
            port=8000,
            transport="streamable-http",
            verbose=True,
            traceback_enabled=True,
            host="localhost",
            queue_enabled=True,
            max_workers_per_agent=2,
            max_queue_size_per_agent=100,
            processing_timeout=30,
            retry_delay=1.0,
            persistence=False,
            max_restart_attempts=5,
            restart_delay=2.0,
            network_monitoring=True,
            max_network_retries=3,
            network_retry_delay=5.0,
            network_timeout=10.0,
            log_level="INFO",
        )
        return aop


def test_aop_initialization_all_parameters():
    """Test AOP initialization with all parameters."""
    with patch(
        "swarms.structs.aop.FastMCP"
    ) as mock_fastmcp_class, patch(
        "swarms.structs.aop.logger"
    ) as mock_logger:
        mock_mcp = Mock()
        mock_mcp.name = "Test AOP"
        mock_mcp.port = 8000
        mock_fastmcp_class.return_value = mock_mcp

        aop = AOP(
            server_name="Test AOP",
            description="Test description",
            agents=None,
            port=9000,
            transport="sse",
            verbose=False,
            traceback_enabled=False,
            host="127.0.0.1",
            queue_enabled=False,
            max_workers_per_agent=5,
            max_queue_size_per_agent=200,
            processing_timeout=60,
            retry_delay=2.0,
            persistence=True,
            max_restart_attempts=10,
            restart_delay=10.0,
            network_monitoring=False,
            max_network_retries=5,
            network_retry_delay=15.0,
            network_timeout=20.0,
            log_level="DEBUG",
        )

        assert aop.server_name == "Test AOP"
        assert aop.description == "Test description"
        assert aop.verbose is False
        assert aop.traceback_enabled is False
        assert aop.host == "127.0.0.1"
        assert aop.port == 9000
        assert aop.transport == "sse"
        assert aop.queue_enabled is False
        assert aop.max_workers_per_agent == 5
        assert aop.max_queue_size_per_agent == 200
        assert aop.processing_timeout == 60
        assert aop.retry_delay == 2.0
        assert aop.persistence is True
        assert aop.max_restart_attempts == 10
        assert aop.restart_delay == 10.0
        assert aop.network_monitoring is False
        assert aop.max_network_retries == 5
        assert aop.network_retry_delay == 15.0
        assert aop.network_timeout == 20.0
        assert aop.log_level == "DEBUG"

        mock_fastmcp_class.assert_called_once()
        mock_logger.remove.assert_called_once()
        mock_logger.add.assert_called_once()


def test_aop_initialization_minimal_parameters():
    """Test AOP initialization with minimal parameters."""
    with patch(
        "swarms.structs.aop.FastMCP"
    ) as mock_fastmcp_class, patch("swarms.structs.aop.logger"):
        mock_mcp = Mock()
        mock_fastmcp_class.return_value = mock_mcp

        aop = AOP()

        assert aop.server_name == "AOP Cluster"
        assert (
            aop.description
            == "A cluster that enables you to deploy multiple agents as tools in an MCP server."
        )
        assert aop.verbose is False
        assert aop.traceback_enabled is True
        assert aop.host == "localhost"
        assert aop.port == 8000
        assert aop.transport == "streamable-http"
        assert aop.queue_enabled is True
        assert aop.max_workers_per_agent == 1
        assert aop.max_queue_size_per_agent == 1000
        assert aop.processing_timeout == 30
        assert aop.retry_delay == 1.0
        assert aop.persistence is False
        assert aop.max_restart_attempts == 10
        assert aop.restart_delay == 5.0
        assert aop.network_monitoring is True
        assert aop.max_network_retries == 5
        assert aop.network_retry_delay == 10.0
        assert aop.network_timeout == 30.0
        assert aop.log_level == "INFO"


def test_aop_initialization_with_agents(
    real_agent, real_agents, mock_fastmcp
):
    """Test AOP initialization with multiple agents."""
    with patch(
        "swarms.structs.aop.FastMCP", return_value=mock_fastmcp
    ), patch("swarms.structs.aop.logger"):
        aop = AOP(agents=real_agents)

        assert len(aop.agents) == 3
        assert "Test-Agent-0" in aop.agents
        assert "Test-Agent-1" in aop.agents
        assert "Test-Agent-2" in aop.agents


def test_add_agent_basic(real_agent, aop_instance, mock_fastmcp):
    """Test basic agent addition."""
    from swarms import Agent

    new_agent = Agent(
        agent_name="new_agent",
        agent_description="New agent description",
        model_name="gpt-3.5-turbo",
        max_loops=1,
        temperature=0.5,
        max_tokens=4096,
    )

    with patch.object(aop_instance, "_register_tool"), patch.object(
        aop_instance, "_register_agent_discovery_tool"
    ):
        result = aop_instance.add_agent(new_agent)

        assert result == "new_agent"
        assert "new_agent" in aop_instance.agents
        assert "new_agent" in aop_instance.tool_configs
        assert "new_agent" in aop_instance.task_queues


def test_add_agent_with_custom_tool_name(
    real_agent, aop_instance, mock_fastmcp
):
    """Test adding agent with custom tool name."""
    from swarms import Agent

    new_agent = Agent(
        agent_name="custom_agent",
        agent_description="Custom agent description",
        model_name="gpt-3.5-turbo",
        max_loops=1,
        temperature=0.5,
        max_tokens=4096,
    )

    with patch.object(aop_instance, "_register_tool"), patch.object(
        aop_instance, "_register_agent_discovery_tool"
    ):
        result = aop_instance.add_agent(
            new_agent, tool_name="custom_tool"
        )

        assert result == "custom_tool"
        assert "custom_tool" in aop_instance.agents
        assert aop_instance.agents["custom_tool"] == new_agent


def test_add_agent_with_custom_schemas(
    real_agent, aop_instance, mock_fastmcp
):
    """Test adding agent with custom input/output schemas."""
    custom_input_schema = {
        "type": "object",
        "properties": {"custom_task": {"type": "string"}},
        "required": ["custom_task"],
    }
    custom_output_schema = {
        "type": "object",
        "properties": {"custom_result": {"type": "string"}},
    }

    from swarms import Agent

    new_agent = Agent(
        agent_name="schema_agent",
        agent_description="Schema test agent",
        model_name="gpt-3.5-turbo",
        max_loops=1,
        temperature=0.5,
        max_tokens=4096,
    )

    with patch.object(aop_instance, "_register_tool"), patch.object(
        aop_instance, "_register_agent_discovery_tool"
    ):
        result = aop_instance.add_agent(
            new_agent,
            input_schema=custom_input_schema,
            output_schema=custom_output_schema,
        )

        config = aop_instance.tool_configs[result]
        assert config.input_schema == custom_input_schema
        assert config.output_schema == custom_output_schema


def test_add_agent_with_all_parameters(
    real_agent, aop_instance, mock_fastmcp
):
    """Test adding agent with all configuration parameters."""
    from swarms import Agent

    new_agent = Agent(
        agent_name="full_config_agent",
        agent_description="Full config test agent",
        model_name="gpt-3.5-turbo",
        max_loops=1,
        temperature=0.5,
        max_tokens=4096,
    )

    with patch.object(aop_instance, "_register_tool"), patch.object(
        aop_instance, "_register_agent_discovery_tool"
    ):
        result = aop_instance.add_agent(
            new_agent,
            tool_name="full_config_tool",
            tool_description="Full config tool description",
            timeout=60,
            max_retries=5,
            verbose=True,
            traceback_enabled=False,
        )

        config = aop_instance.tool_configs[result]
        assert config.tool_name == "full_config_tool"
        assert (
            config.tool_description == "Full config tool description"
        )
        assert config.timeout == 60
        assert config.max_retries == 5
        assert config.verbose is True
        assert config.traceback_enabled is False


def test_add_agent_none_agent(aop_instance, mock_fastmcp):
    """Test adding None agent raises ValueError."""
    with patch.object(aop_instance, "_register_tool"), patch.object(
        aop_instance, "_register_agent_discovery_tool"
    ):
        with pytest.raises(ValueError, match="Agent cannot be None"):
            aop_instance.add_agent(None)


def test_add_agent_duplicate_tool_name(
    real_agent, aop_instance, mock_fastmcp
):
    """Test adding agent with duplicate tool name raises ValueError."""
    from swarms import Agent

    new_agent = Agent(
        agent_name="duplicate_agent",
        agent_description="Duplicate agent",
        model_name="gpt-3.5-turbo",
        max_loops=1,
        temperature=0.5,
        max_tokens=4096,
    )

    with patch.object(aop_instance, "_register_tool"), patch.object(
        aop_instance, "_register_agent_discovery_tool"
    ):
        aop_instance.add_agent(new_agent, tool_name="test_agent")

        with pytest.raises(
            ValueError, match="Tool name 'test_agent' already exists"
        ):
            aop_instance.add_agent(new_agent, tool_name="test_agent")


def test_add_agents_batch_basic(
    real_agents, aop_instance, mock_fastmcp
):
    """Test adding multiple agents in batch."""
    with patch.object(
        aop_instance, "add_agent"
    ) as mock_add_agent, patch.object(
        aop_instance, "_register_agent_discovery_tool"
    ):
        mock_add_agent.return_value = "test_tool"

        result = aop_instance.add_agents_batch(real_agents)

        assert len(result) == 3
        assert mock_add_agent.call_count == 3
        mock_add_agent.assert_any_call(
            real_agents[0], None, None, None, None, 30, 3, None, None
        )
        mock_add_agent.assert_any_call(
            real_agents[1], None, None, None, None, 30, 3, None, None
        )
        mock_add_agent.assert_any_call(
            real_agents[2], None, None, None, None, 30, 3, None, None
        )


def test_add_agents_batch_with_custom_parameters(
    real_agents, aop_instance, mock_fastmcp
):
    """Test adding multiple agents in batch with custom parameters."""
    tool_names = ["tool1", "tool2", "tool3"]
    timeouts = [60, 45, 30]
    verbose_list = [True, False, True]

    with patch.object(
        aop_instance, "add_agent"
    ) as mock_add_agent, patch.object(
        aop_instance, "_register_agent_discovery_tool"
    ):
        mock_add_agent.return_value = "test_tool"

        result = aop_instance.add_agents_batch(
            real_agents,
            tool_names=tool_names,
            timeouts=timeouts,
            verbose_list=verbose_list,
        )

        assert len(result) == 3
        assert mock_add_agent.call_count == 3
        mock_add_agent.assert_any_call(
            real_agents[0],
            "tool1",
            None,
            None,
            None,
            60,
            3,
            True,
            None,
        )
        mock_add_agent.assert_any_call(
            real_agents[1],
            "tool2",
            None,
            None,
            None,
            45,
            3,
            False,
            None,
        )
        mock_add_agent.assert_any_call(
            real_agents[2],
            "tool3",
            None,
            None,
            None,
            30,
            3,
            True,
            None,
        )


def test_add_agents_batch_empty_list(
    real_agents, aop_instance, mock_fastmcp
):
    """Test adding empty agents list raises ValueError."""
    with patch.object(aop_instance, "_register_agent_discovery_tool"):
        with pytest.raises(
            ValueError, match="Agents list cannot be empty"
        ):
            aop_instance.add_agents_batch([])


def test_add_agents_batch_with_none_values(
    real_agents, aop_instance, mock_fastmcp
):
    """Test adding agents list with None values raises ValueError."""
    agents_with_none = real_agents.copy()
    agents_with_none.append(None)

    with patch.object(aop_instance, "_register_agent_discovery_tool"):
        with pytest.raises(
            ValueError, match="Agents list cannot contain None values"
        ):
            aop_instance.add_agents_batch(agents_with_none)


def test_remove_agent_existing(aop_instance, mock_fastmcp):
    """Test removing existing agent."""
    with patch.object(
        aop_instance.task_queues["test_agent"], "stop_workers"
    ):
        result = aop_instance.remove_agent("test_agent")

        assert result is True
        assert "test_agent" not in aop_instance.agents
        assert "test_agent" not in aop_instance.tool_configs
        assert "test_agent" not in aop_instance.task_queues


def test_remove_agent_nonexistent(aop_instance, mock_fastmcp):
    """Test removing non-existent agent."""
    result = aop_instance.remove_agent("nonexistent_agent")

    assert result is False


def test_list_agents(aop_instance, mock_fastmcp):
    """Test listing agents."""
    result = aop_instance.list_agents()

    assert isinstance(result, list)
    assert "test_agent" in result


def test_get_agent_info_existing(aop_instance, mock_fastmcp):
    """Test getting info for existing agent."""
    result = aop_instance.get_agent_info("test_agent")

    assert result is not None
    assert result["tool_name"] == "test_agent"
    assert result["agent_name"] == "test_agent"
    assert result["agent_description"] == "Test agent description"
    assert result["model_name"] == "gpt-3.5-turbo"


def test_get_agent_info_nonexistent(aop_instance, mock_fastmcp):
    """Test getting info for non-existent agent."""
    result = aop_instance.get_agent_info("nonexistent_agent")

    assert result is None


def test_get_queue_stats_queue_enabled(aop_instance, mock_fastmcp):
    """Test getting queue stats when queue is enabled."""
    # Mock the task queue stats
    mock_stats = Mock()
    mock_stats.total_tasks = 10
    mock_stats.completed_tasks = 8
    mock_stats.failed_tasks = 1
    mock_stats.pending_tasks = 1
    mock_stats.processing_tasks = 0
    mock_stats.average_processing_time = 2.5
    mock_stats.queue_size = 1

    with patch.object(
        aop_instance.task_queues["test_agent"],
        "get_stats",
        return_value=mock_stats,
    ), patch.object(
        aop_instance.task_queues["test_agent"],
        "get_status",
        return_value=QueueStatus.RUNNING,
    ):
        result = aop_instance.get_queue_stats("test_agent")

        assert result["success"] is True
        assert result["agent_name"] == "test_agent"
        assert result["stats"]["total_tasks"] == 10
        assert result["stats"]["completed_tasks"] == 8
        assert result["stats"]["failed_tasks"] == 1
        assert result["stats"]["pending_tasks"] == 1
        assert result["stats"]["average_processing_time"] == 2.5
        assert result["stats"]["queue_size"] == 1
        assert result["stats"]["queue_status"] == "running"


def test_get_queue_stats_queue_disabled(aop_instance, mock_fastmcp):
    """Test getting queue stats when queue is disabled."""
    aop_instance.queue_enabled = False

    result = aop_instance.get_queue_stats("test_agent")

    assert result["success"] is False
    assert result["error"] == "Queue system is not enabled"
    assert result["stats"] == {}


def test_get_queue_stats_nonexistent_agent(
    aop_instance, mock_fastmcp
):
    """Test getting queue stats for non-existent agent."""
    result = aop_instance.get_queue_stats("nonexistent_agent")

    assert result["success"] is False
    assert (
        result["error"]
        == "Agent 'nonexistent_agent' not found or has no queue"
    )


def test_get_queue_stats_all_agents(aop_instance, mock_fastmcp):
    """Test getting queue stats for all agents."""
    # Mock stats for both agents
    mock_stats1 = Mock()
    mock_stats1.total_tasks = 10
    mock_stats1.completed_tasks = 8
    mock_stats1.failed_tasks = 1
    mock_stats1.pending_tasks = 1
    mock_stats1.processing_tasks = 0
    mock_stats1.average_processing_time = 2.5
    mock_stats1.queue_size = 1

    mock_stats2 = Mock()
    mock_stats2.total_tasks = 5
    mock_stats2.completed_tasks = 5
    mock_stats2.failed_tasks = 0
    mock_stats2.pending_tasks = 0
    mock_stats2.processing_tasks = 0
    mock_stats2.average_processing_time = 1.8
    mock_stats2.queue_size = 0

    with patch.object(
        aop_instance.task_queues["test_agent"],
        "get_stats",
        return_value=mock_stats1,
    ), patch.object(
        aop_instance.task_queues["test_agent"],
        "get_status",
        return_value=QueueStatus.RUNNING,
    ):
        result = aop_instance.get_queue_stats()

        assert result["success"] is True
        assert result["total_agents"] == 1
        assert "test_agent" in result["stats"]
        assert result["stats"]["test_agent"]["total_tasks"] == 10


def test_pause_agent_queue_success(aop_instance, mock_fastmcp):
    """Test pausing agent queue successfully."""
    with patch.object(
        aop_instance.task_queues["test_agent"], "pause_workers"
    ) as mock_pause:
        result = aop_instance.pause_agent_queue("test_agent")

        assert result is True
        mock_pause.assert_called_once()


def test_pause_agent_queue_disabled(aop_instance, mock_fastmcp):
    """Test pausing agent queue when queue system disabled."""
    aop_instance.queue_enabled = False

    result = aop_instance.pause_agent_queue("test_agent")

    assert result is False


def test_pause_agent_queue_nonexistent(aop_instance, mock_fastmcp):
    """Test pausing non-existent agent queue."""
    result = aop_instance.pause_agent_queue("nonexistent_agent")

    assert result is False


def test_resume_agent_queue_success(aop_instance, mock_fastmcp):
    """Test resuming agent queue successfully."""
    with patch.object(
        aop_instance.task_queues["test_agent"], "resume_workers"
    ) as mock_resume:
        result = aop_instance.resume_agent_queue("test_agent")

        assert result is True
        mock_resume.assert_called_once()


def test_clear_agent_queue_success(aop_instance, mock_fastmcp):
    """Test clearing agent queue successfully."""
    with patch.object(
        aop_instance.task_queues["test_agent"],
        "clear_queue",
        return_value=5,
    ) as mock_clear:
        result = aop_instance.clear_agent_queue("test_agent")

        assert result == 5
        mock_clear.assert_called_once()


def test_clear_agent_queue_disabled(aop_instance, mock_fastmcp):
    """Test clearing agent queue when queue system disabled."""
    aop_instance.queue_enabled = False

    result = aop_instance.clear_agent_queue("test_agent")

    assert result == -1


def test_get_task_status_success(aop_instance, mock_fastmcp):
    """Test getting task status successfully."""
    mock_task = Mock()
    mock_task.task_id = "test_task_id"
    mock_task.status = TaskStatus.COMPLETED
    mock_task.created_at = 1234567890.0
    mock_task.result = "Task completed"
    mock_task.error = None
    mock_task.retry_count = 0
    mock_task.max_retries = 3
    mock_task.priority = 0

    with patch.object(
        aop_instance.task_queues["test_agent"],
        "get_task",
        return_value=mock_task,
    ):
        result = aop_instance.get_task_status(
            "test_agent", "test_task_id"
        )

        assert result["success"] is True
        assert result["task"]["task_id"] == "test_task_id"
        assert result["task"]["status"] == "completed"
        assert result["task"]["result"] == "Task completed"


def test_get_task_status_queue_disabled(aop_instance, mock_fastmcp):
    """Test getting task status when queue disabled."""
    aop_instance.queue_enabled = False

    result = aop_instance.get_task_status(
        "test_agent", "test_task_id"
    )

    assert result["success"] is False
    assert result["error"] == "Queue system is not enabled"


def test_get_task_status_task_not_found(aop_instance, mock_fastmcp):
    """Test getting status for non-existent task."""
    with patch.object(
        aop_instance.task_queues["test_agent"],
        "get_task",
        return_value=None,
    ):
        result = aop_instance.get_task_status(
            "test_agent", "test_task_id"
        )

        assert result["success"] is False
        assert result["error"] == "Task 'test_task_id' not found"


def test_cancel_task_success(aop_instance, mock_fastmcp):
    """Test cancelling task successfully."""
    with patch.object(
        aop_instance.task_queues["test_agent"],
        "cancel_task",
        return_value=True,
    ):
        result = aop_instance.cancel_task(
            "test_agent", "test_task_id"
        )

        assert result is True


def test_cancel_task_queue_disabled(aop_instance, mock_fastmcp):
    """Test cancelling task when queue disabled."""
    aop_instance.queue_enabled = False

    result = aop_instance.cancel_task("test_agent", "test_task_id")

    assert result is False


def test_pause_all_queues(aop_instance, mock_fastmcp):
    """Test pausing all queues."""
    with patch.object(
        aop_instance, "pause_agent_queue", return_value=True
    ) as mock_pause:
        result = aop_instance.pause_all_queues()

        assert len(result) == 1
        assert result["test_agent"] is True
        mock_pause.assert_called_once_with("test_agent")


def test_pause_all_queues_disabled(aop_instance, mock_fastmcp):
    """Test pausing all queues when disabled."""
    aop_instance.queue_enabled = False

    result = aop_instance.pause_all_queues()

    assert result == {}


def test_resume_all_queues(aop_instance, mock_fastmcp):
    """Test resuming all queues."""
    with patch.object(
        aop_instance, "resume_agent_queue", return_value=True
    ) as mock_resume:
        result = aop_instance.resume_all_queues()

        assert len(result) == 1
        assert result["test_agent"] is True
        mock_resume.assert_called_once_with("test_agent")


def test_clear_all_queues(aop_instance, mock_fastmcp):
    """Test clearing all queues."""
    with patch.object(
        aop_instance, "clear_agent_queue", return_value=5
    ) as mock_clear:
        result = aop_instance.clear_all_queues()

        assert len(result) == 1
        assert result["test_agent"] == 5
        mock_clear.assert_called_once_with("test_agent")


def test_enable_persistence(aop_instance, mock_fastmcp):
    """Test enabling persistence mode."""
    aop_instance.enable_persistence()

    assert aop_instance._persistence_enabled is True


def test_disable_persistence(aop_instance, mock_fastmcp):
    """Test disabling persistence mode."""
    aop_instance.disable_persistence()

    assert aop_instance._persistence_enabled is False
    assert aop_instance._shutdown_requested is True


def test_request_shutdown(aop_instance, mock_fastmcp):
    """Test requesting server shutdown."""
    aop_instance.request_shutdown()

    assert aop_instance._shutdown_requested is True


def test_get_persistence_status(aop_instance, mock_fastmcp):
    """Test getting persistence status."""
    result = aop_instance.get_persistence_status()

    assert result["persistence_enabled"] is False
    assert result["shutdown_requested"] is False
    assert result["restart_count"] == 0
    assert result["max_restart_attempts"] == 10
    assert result["restart_delay"] == 5.0
    assert result["remaining_restarts"] == 10


def test_reset_restart_count(aop_instance, mock_fastmcp):
    """Test resetting restart counter."""
    aop_instance._restart_count = 5

    aop_instance.reset_restart_count()

    assert aop_instance._restart_count == 0


def test_get_network_status(aop_instance, mock_fastmcp):
    """Test getting network status."""
    result = aop_instance.get_network_status()

    assert result["network_monitoring_enabled"] is True
    assert result["network_connected"] is True
    assert result["network_retry_count"] == 0
    assert result["max_network_retries"] == 5
    assert result["network_retry_delay"] == 10.0
    assert result["network_timeout"] == 30.0
    assert result["last_network_error"] is None
    assert result["remaining_network_retries"] == 5
    assert result["host"] == "localhost"
    assert result["port"] == 8000


def test_reset_network_retry_count(aop_instance, mock_fastmcp):
    """Test resetting network retry counter."""
    aop_instance._network_retry_count = 3
    aop_instance._last_network_error = "Test error"
    aop_instance._network_connected = False

    aop_instance.reset_network_retry_count()

    assert aop_instance._network_retry_count == 0
    assert aop_instance._last_network_error is None
    assert aop_instance._network_connected is True


def test_get_server_info(aop_instance, mock_fastmcp):
    """Test getting comprehensive server information."""
    result = aop_instance.get_server_info()

    assert result["server_name"] == "Test AOP"
    assert result["description"] == "Test AOP description"
    assert result["total_tools"] == 1
    assert result["tools"] == ["test_agent"]
    assert result["verbose"] is True
    assert result["traceback_enabled"] is True
    assert result["log_level"] == "INFO"
    assert result["transport"] == "streamable-http"
    assert result["queue_enabled"] is True
    assert "persistence" in result
    assert "network" in result
    assert "tool_details" in result
    assert "queue_config" in result


def test_is_network_error_connection_error(
    aop_instance, mock_fastmcp
):
    """Test detecting connection errors as network errors."""
    error = ConnectionError("Connection failed")

    result = aop_instance._is_network_error(error)

    assert result is True


def test_is_network_error_timeout_error(aop_instance, mock_fastmcp):
    """Test detecting timeout errors as network errors."""
    error = TimeoutError("Request timed out")

    result = aop_instance._is_network_error(error)

    assert result is True


def test_is_network_error_socket_error(aop_instance, mock_fastmcp):
    """Test detecting socket errors as network errors."""
    error = socket.gaierror("Name resolution failed")

    result = aop_instance._is_network_error(error)

    assert result is True


def test_is_network_error_non_network_error(
    aop_instance, mock_fastmcp
):
    """Test non-network errors are not detected as network errors."""
    error = ValueError("Invalid value")

    result = aop_instance._is_network_error(error)

    assert result is False


def test_is_network_error_by_message(aop_instance, mock_fastmcp):
    """Test detecting network errors by error message content."""
    error = Exception("Connection refused by server")

    result = aop_instance._is_network_error(error)

    assert result is True


def test_get_network_error_message_connection_refused(
    aop_instance, mock_fastmcp
):
    """Test getting custom message for connection refused errors."""
    error = ConnectionRefusedError("Connection refused")

    result = aop_instance._get_network_error_message(error, 1)

    assert "NETWORK ERROR: Connection refused" in result
    assert "attempt 1/5" in result


def test_get_network_error_message_timeout(
    aop_instance, mock_fastmcp
):
    """Test getting custom message for timeout errors."""
    error = TimeoutError("Request timed out")

    result = aop_instance._get_network_error_message(error, 2)

    assert "NETWORK ERROR: Connection timeout" in result
    assert "attempt 2/5" in result


def test_test_network_connectivity_success(
    aop_instance, mock_fastmcp
):
    """Test network connectivity test success."""
    with patch(
        "socket.gethostbyname", return_value="127.0.0.1"
    ), patch("socket.socket") as mock_socket_class:
        mock_socket = Mock()
        mock_socket.connect_ex.return_value = 0
        mock_socket_class.return_value = mock_socket

        result = aop_instance._test_network_connectivity()

        assert result is True
        mock_socket.connect_ex.assert_called_once_with(
            ("localhost", 8000)
        )


def test_test_network_connectivity_failure(
    aop_instance, mock_fastmcp
):
    """Test network connectivity test failure."""
    with patch(
        "socket.gethostbyname",
        side_effect=socket.gaierror("Name resolution failed"),
    ):
        result = aop_instance._test_network_connectivity()

        assert result is False


def test_handle_network_error_first_retry(aop_instance, mock_fastmcp):
    """Test handling network error on first retry."""
    error = ConnectionError("Connection failed")

    with patch.object(
        aop_instance, "_test_network_connectivity", return_value=True
    ) as mock_test:
        result = aop_instance._handle_network_error(error)

        assert result is True
        assert aop_instance._network_retry_count == 1
        assert aop_instance._network_connected is True
        mock_test.assert_called_once()


def test_handle_network_error_max_retries(aop_instance, mock_fastmcp):
    """Test handling network error at max retries."""
    error = ConnectionError("Connection failed")
    aop_instance._network_retry_count = 5  # Max retries

    with patch.object(
        aop_instance, "_test_network_connectivity", return_value=False
    ):
        result = aop_instance._handle_network_error(error)

        assert result is False
        assert aop_instance._network_retry_count == 6


def test_handle_network_error_disabled_monitoring(
    aop_instance, mock_fastmcp
):
    """Test handling network error when monitoring disabled."""
    aop_instance.network_monitoring = False
    error = ConnectionError("Connection failed")

    result = aop_instance._handle_network_error(error)

    assert result is False
    assert aop_instance._network_retry_count == 0


def test_start_server_basic(aop_instance, mock_fastmcp):
    """Test starting server with basic configuration."""
    with patch("swarms.structs.aop.logger") as mock_logger:
        aop_instance.start_server()

        mock_logger.info.assert_any_call(
            "Starting MCP server 'Test AOP' on localhost:8000\n"
            "Transport: streamable-http\n"
            "Log level: INFO\n"
            "Verbose mode: True\n"
            "Traceback enabled: True\n"
            "Queue enabled: True\n"
            "Available tools: ['test_agent']"
        )


def test_start_server_keyboard_interrupt(aop_instance, mock_fastmcp):
    """Test starting server with keyboard interrupt."""
    mock_fastmcp.run.side_effect = KeyboardInterrupt()

    with patch("swarms.structs.aop.logger"):
        aop_instance.start_server()

        # Should handle KeyboardInterrupt gracefully
        assert True  # If we get here, the test passed


def test_run_without_persistence(aop_instance, mock_fastmcp):
    """Test running server without persistence."""
    with patch.object(aop_instance, "start_server") as mock_start:
        aop_instance.run()

        mock_start.assert_called_once()


def test_run_with_persistence_success(aop_instance, mock_fastmcp):
    """Test running server with persistence on successful execution."""
    aop_instance._persistence_enabled = True

    with patch.object(aop_instance, "start_server") as mock_start:
        aop_instance.run()

        mock_start.assert_called_once()
        assert aop_instance._restart_count == 0


def test_run_with_persistence_keyboard_interrupt(
    aop_instance, mock_fastmcp
):
    """Test running server with persistence on keyboard interrupt."""
    aop_instance._persistence_enabled = True
    aop_instance._shutdown_requested = False

    with patch.object(
        aop_instance, "start_server", side_effect=KeyboardInterrupt()
    ) as mock_start:
        aop_instance.run()

        mock_start.assert_called_once()
        assert aop_instance._restart_count == 1


def test_run_with_persistence_network_error(
    aop_instance, mock_fastmcp
):
    """Test running server with persistence on network error."""
    aop_instance._persistence_enabled = True
    aop_instance._shutdown_requested = False

    network_error = ConnectionError("Connection failed")
    with patch.object(
        aop_instance, "start_server", side_effect=network_error
    ) as mock_start, patch.object(
        aop_instance, "_is_network_error", return_value=True
    ) as mock_is_network, patch.object(
        aop_instance, "_handle_network_error", return_value=True
    ) as mock_handle_network:
        aop_instance.run()

        mock_start.assert_called_once()
        mock_is_network.assert_called_once_with(network_error)
        mock_handle_network.assert_called_once_with(network_error)
        assert aop_instance._restart_count == 1


def test_run_with_persistence_non_network_error(
    aop_instance, mock_fastmcp
):
    """Test running server with persistence on non-network error."""
    aop_instance._persistence_enabled = True
    aop_instance._shutdown_requested = False

    error = ValueError("Invalid configuration")
    with patch.object(
        aop_instance, "start_server", side_effect=error
    ) as mock_start, patch.object(
        aop_instance, "_is_network_error", return_value=False
    ) as mock_is_network:
        aop_instance.run()

        mock_start.assert_called_once()
        mock_is_network.assert_called_once_with(error)
        assert aop_instance._restart_count == 1


def test_run_with_persistence_max_restarts(
    aop_instance, mock_fastmcp
):
    """Test running server with persistence hitting max restarts."""
    aop_instance._persistence_enabled = True
    aop_instance._shutdown_requested = False
    aop_instance._restart_count = 10  # At max
    aop_instance.max_restart_attempts = 10

    error = ValueError("Invalid configuration")
    with patch.object(
        aop_instance, "start_server", side_effect=error
    ), patch.object(
        aop_instance, "_is_network_error", return_value=False
    ):
        aop_instance.run()

        assert aop_instance._restart_count == 11


def test_aop_cluster_initialization():
    """Test AOPCluster initialization."""
    urls = ["http://localhost:8000", "http://localhost:8001"]

    cluster = AOPCluster(urls, transport="sse")

    assert cluster.urls == urls
    assert cluster.transport == "sse"


def test_aop_cluster_get_tools():
    """Test AOPCluster getting tools from servers."""
    urls = ["http://localhost:8000"]

    with patch(
        "swarms.structs.aop.get_tools_for_multiple_mcp_servers"
    ) as mock_get_tools:
        mock_get_tools.return_value = [{"test": "data"}]

        cluster = AOPCluster(urls)
        result = cluster.get_tools(output_type="dict")

        assert result == [{"test": "data"}]
        mock_get_tools.assert_called_once_with(
            urls=urls,
            format="openai",
            output_type="dict",
            transport="streamable-http",
        )


def test_aop_cluster_find_tool_by_server_name():
    """Test AOPCluster finding tool by server name."""
    urls = ["http://localhost:8000"]

    mock_tools = [
        {"function": {"name": "test_agent"}},
        {"function": {"name": "other_agent"}},
    ]

    with patch.object(
        AOPCluster, "get_tools", return_value=mock_tools
    ) as mock_get_tools:
        cluster = AOPCluster(urls)
        result = cluster.find_tool_by_server_name("test_agent")

        assert result == mock_tools[0]
        mock_get_tools.assert_called_once()


def test_aop_cluster_find_tool_not_found():
    """Test AOPCluster finding non-existent tool."""
    urls = ["http://localhost:8000"]

    mock_tools = [{"function": {"name": "other_agent"}}]

    with patch.object(
        AOPCluster, "get_tools", return_value=mock_tools
    ):
        cluster = AOPCluster(urls)
        result = cluster.find_tool_by_server_name("nonexistent_agent")

        assert result is None


def test_aop_initialization_with_queue_disabled(
    real_agent, mock_fastmcp
):
    """Test AOP initialization with queue system disabled."""
    with patch(
        "swarms.structs.aop.FastMCP", return_value=mock_fastmcp
    ), patch("swarms.structs.aop.logger"):
        aop = AOP(queue_enabled=False)

        assert aop.queue_enabled is False
        assert len(aop.task_queues) == 0


def test_aop_initialization_edge_cases(mock_fastmcp):
    """Test AOP initialization with edge case values."""
    with patch(
        "swarms.structs.aop.FastMCP", return_value=mock_fastmcp
    ), patch("swarms.structs.aop.logger"):
        aop = AOP(
            server_name="",
            port=0,
            max_workers_per_agent=0,
            max_queue_size_per_agent=0,
            max_restart_attempts=0,
            max_network_retries=0,
        )

        assert aop.server_name == ""
        assert aop.port == 0
        assert aop.max_workers_per_agent == 0
        assert aop.max_queue_size_per_agent == 0
        assert aop.max_restart_attempts == 0
        assert aop.max_network_retries == 0


def test_add_agent_queue_disabled(
    real_agent, aop_instance, mock_fastmcp
):
    """Test adding agent when queue is disabled."""
    aop_instance.queue_enabled = False

    from swarms import Agent

    new_agent = Agent(
        agent_name="no_queue_agent",
        agent_description="No queue agent",
        model_name="gpt-3.5-turbo",
        max_loops=1,
        temperature=0.5,
        max_tokens=4096,
    )

    with patch.object(aop_instance, "_register_tool"), patch.object(
        aop_instance, "_register_agent_discovery_tool"
    ):
        result = aop_instance.add_agent(new_agent)

        assert result == "no_queue_agent"
        assert "no_queue_agent" not in aop_instance.task_queues


def test_get_queue_stats_all_agents_comprehensive(
    aop_instance, mock_fastmcp
):
    """Test getting comprehensive queue stats for all agents."""
    # Add another agent for testing
    new_agent = Mock(spec=Agent)
    new_agent.agent_name = "second_agent"
    new_agent.run = Mock(return_value="Second response")

    with patch.object(aop_instance, "_register_tool"), patch.object(
        aop_instance, "_register_agent_discovery_tool"
    ):
        aop_instance.add_agent(new_agent, tool_name="second_agent")

    # Mock stats for both agents
    mock_stats1 = Mock()
    mock_stats1.total_tasks = 10
    mock_stats1.completed_tasks = 8
    mock_stats1.failed_tasks = 1
    mock_stats1.pending_tasks = 1
    mock_stats1.processing_tasks = 0
    mock_stats1.average_processing_time = 2.5
    mock_stats1.queue_size = 1

    mock_stats2 = Mock()
    mock_stats2.total_tasks = 5
    mock_stats2.completed_tasks = 5
    mock_stats2.failed_tasks = 0
    mock_stats2.pending_tasks = 0
    mock_stats2.processing_tasks = 0
    mock_stats2.average_processing_time = 1.8
    mock_stats2.queue_size = 0

    def mock_get_stats(tool_name):
        if tool_name == "test_agent":
            return mock_stats1
        elif tool_name == "second_agent":
            return mock_stats2
        return Mock()

    def mock_get_status(tool_name):
        return QueueStatus.RUNNING

    with patch.object(
        aop_instance.task_queues["test_agent"],
        "get_stats",
        mock_get_stats,
    ), patch.object(
        aop_instance.task_queues["second_agent"],
        "get_stats",
        mock_get_stats,
    ), patch.object(
        aop_instance.task_queues["test_agent"],
        "get_status",
        mock_get_status,
    ), patch.object(
        aop_instance.task_queues["second_agent"],
        "get_status",
        mock_get_status,
    ):
        result = aop_instance.get_queue_stats()

        assert result["success"] is True
        assert result["total_agents"] == 2
        assert "test_agent" in result["stats"]
        assert "second_agent" in result["stats"]
        assert result["stats"]["test_agent"]["total_tasks"] == 10
        assert result["stats"]["second_agent"]["total_tasks"] == 5
