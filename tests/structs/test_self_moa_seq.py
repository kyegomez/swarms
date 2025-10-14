from unittest.mock import Mock, patch

import pytest

from swarms.structs.self_moa_seq import SelfMoASeq

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def basic_seq():
    """Create a basic SelfMoASeq instance for testing."""
    return SelfMoASeq(
        num_samples=3,
        window_size=4,
        reserved_slots=2,
        max_iterations=5,
        verbose=False,
        enable_logging=False,
    )


@pytest.fixture
def custom_retry_seq():
    """Create a SelfMoASeq instance with custom retry parameters."""
    return SelfMoASeq(
        num_samples=2,
        max_retries=5,
        retry_delay=0.5,
        retry_backoff_multiplier=1.5,
        retry_max_delay=10.0,
        verbose=False,
        enable_logging=False,
    )


@pytest.fixture
def mock_agents():
    """Create mock agents for testing."""
    proposer = Mock()
    aggregator = Mock()
    return proposer, aggregator


# ============================================================================
# Initialization and Parameter Validation Tests
# ============================================================================


def test_default_initialization():
    """Test that SelfMoASeq initializes with default parameters."""
    seq = SelfMoASeq()

    assert seq.model_name == "gpt-4o-mini"
    assert seq.temperature == 0.7
    assert seq.window_size == 6
    assert seq.reserved_slots == 3
    assert seq.max_iterations == 10
    assert seq.max_tokens == 2000
    assert seq.num_samples == 30
    assert seq.enable_logging is True
    assert seq.log_level == "INFO"
    assert seq.verbose is True
    assert seq.max_retries == 3
    assert seq.retry_delay == 1.0
    assert seq.retry_backoff_multiplier == 2.0
    assert seq.retry_max_delay == 60.0


def test_custom_initialization():
    """Test initialization with custom parameters."""
    seq = SelfMoASeq(
        model_name="custom-model",
        temperature=0.5,
        window_size=8,
        reserved_slots=2,
        max_iterations=15,
        max_tokens=3000,
        num_samples=20,
        enable_logging=False,
        log_level="DEBUG",
        verbose=False,
        proposer_model_name="proposer-model",
        aggregator_model_name="aggregator-model",
        max_retries=5,
        retry_delay=2.0,
        retry_backoff_multiplier=3.0,
        retry_max_delay=120.0,
    )

    assert seq.model_name == "custom-model"
    assert seq.temperature == 0.5
    assert seq.window_size == 8
    assert seq.reserved_slots == 2
    assert seq.max_iterations == 15
    assert seq.max_tokens == 3000
    assert seq.num_samples == 20
    assert seq.enable_logging is False
    assert seq.log_level == "DEBUG"
    assert seq.verbose is False
    assert seq.max_retries == 5
    assert seq.retry_delay == 2.0
    assert seq.retry_backoff_multiplier == 3.0
    assert seq.retry_max_delay == 120.0


def test_window_size_validation():
    """Test window_size parameter validation."""
    # Valid window_size
    seq = SelfMoASeq(window_size=2)
    assert seq.window_size == 2

    # Invalid window_size
    with pytest.raises(
        ValueError, match="window_size must be at least 2"
    ):
        SelfMoASeq(window_size=1)


def test_reserved_slots_validation():
    """Test reserved_slots parameter validation."""
    # Valid reserved_slots
    seq = SelfMoASeq(window_size=6, reserved_slots=3)
    assert seq.reserved_slots == 3

    # Invalid reserved_slots (>= window_size)
    with pytest.raises(
        ValueError,
        match="reserved_slots must be less than window_size",
    ):
        SelfMoASeq(window_size=6, reserved_slots=6)


def test_temperature_validation():
    """Test temperature parameter validation."""
    # Valid temperature
    seq = SelfMoASeq(temperature=1.5)
    assert seq.temperature == 1.5

    # Invalid temperature (too high)
    with pytest.raises(
        ValueError, match="temperature must be between 0 and 2"
    ):
        SelfMoASeq(temperature=2.5)

    # Invalid temperature (negative)
    with pytest.raises(
        ValueError, match="temperature must be between 0 and 2"
    ):
        SelfMoASeq(temperature=-0.1)


def test_max_iterations_validation():
    """Test max_iterations parameter validation."""
    # Valid max_iterations
    seq = SelfMoASeq(max_iterations=5)
    assert seq.max_iterations == 5

    # Invalid max_iterations
    with pytest.raises(
        ValueError, match="max_iterations must be at least 1"
    ):
        SelfMoASeq(max_iterations=0)


def test_num_samples_validation():
    """Test num_samples parameter validation."""
    # Valid num_samples
    seq = SelfMoASeq(num_samples=5)
    assert seq.num_samples == 5

    # Invalid num_samples
    with pytest.raises(
        ValueError, match="num_samples must be at least 2"
    ):
        SelfMoASeq(num_samples=1)


def test_retry_parameters_validation():
    """Test retry parameters validation."""
    # Valid retry parameters
    seq = SelfMoASeq(
        max_retries=5,
        retry_delay=2.0,
        retry_backoff_multiplier=1.5,
        retry_max_delay=10.0,
    )
    assert seq.max_retries == 5
    assert seq.retry_delay == 2.0
    assert seq.retry_backoff_multiplier == 1.5
    assert seq.retry_max_delay == 10.0

    # Invalid max_retries
    with pytest.raises(
        ValueError, match="max_retries must be non-negative"
    ):
        SelfMoASeq(max_retries=-1)

    # Invalid retry_delay
    with pytest.raises(
        ValueError, match="retry_delay must be non-negative"
    ):
        SelfMoASeq(retry_delay=-1.0)

    # Invalid retry_backoff_multiplier
    with pytest.raises(
        ValueError, match="retry_backoff_multiplier must be >= 1"
    ):
        SelfMoASeq(retry_backoff_multiplier=0.5)

    # Invalid retry_max_delay
    with pytest.raises(
        ValueError, match="retry_max_delay must be >= retry_delay"
    ):
        SelfMoASeq(retry_delay=10.0, retry_max_delay=5.0)


# ============================================================================
# Retry Functionality Tests
# ============================================================================


def test_retry_decorator_property(basic_seq):
    """Test that retry decorator property works correctly."""
    decorator = basic_seq.retry_decorator
    assert callable(decorator)


def test_get_retry_decorator(basic_seq):
    """Test _get_retry_decorator method."""
    decorator = basic_seq._get_retry_decorator()
    assert callable(decorator)


def test_retry_configuration_inheritance(custom_retry_seq):
    """Test that retry configuration is properly inherited."""
    assert custom_retry_seq.max_retries == 5
    assert custom_retry_seq.retry_delay == 0.5
    assert custom_retry_seq.retry_backoff_multiplier == 1.5
    assert custom_retry_seq.retry_max_delay == 10.0


def test_retry_functionality_with_mock(custom_retry_seq, mock_agents):
    """Test retry functionality with mocked agents."""
    proposer, aggregator = mock_agents

    # Configure mock to fail first time, succeed second time
    proposer.run.side_effect = [
        Exception("Simulated failure"),
        "Sample 1",
        "Sample 2",
    ]

    aggregator.run.side_effect = [
        Exception("Simulated aggregation failure"),
        "Aggregated result",
    ]

    # Patch the agents
    with patch.object(
        custom_retry_seq, "proposer", proposer
    ), patch.object(custom_retry_seq, "aggregator", aggregator):

        # This should succeed after retries
        result = custom_retry_seq.run("Test task")

        assert result is not None
        assert "final_output" in result
        assert "all_samples" in result
        assert "metrics" in result


# ============================================================================
# Core Methods Tests
# ============================================================================


def test_generate_samples_success(basic_seq, mock_agents):
    """Test successful sample generation."""
    proposer, _ = mock_agents
    proposer.run.return_value = "Generated sample"

    with patch.object(basic_seq, "proposer", proposer):
        samples = basic_seq._generate_samples("Test task", 3)

        assert len(samples) == 3
        assert all(sample == "Generated sample" for sample in samples)
        assert basic_seq.metrics["total_samples_generated"] == 3


def test_generate_samples_with_retry(basic_seq, mock_agents):
    """Test sample generation with retry on failure."""
    proposer, _ = mock_agents
    proposer.run.side_effect = [
        Exception("First failure"),
        Exception("Second failure"),
        "Sample 1",
        "Sample 2",
        "Sample 3",
    ]

    with patch.object(basic_seq, "proposer", proposer):
        samples = basic_seq._generate_samples("Test task", 3)

        assert len(samples) == 3
        assert basic_seq.metrics["total_samples_generated"] == 3


def test_format_aggregation_prompt(basic_seq):
    """Test aggregation prompt formatting."""
    task = "Test task"
    samples = ["Sample 1", "Sample 2", "Sample 3"]
    best_so_far = "Best response"

    prompt = basic_seq._format_aggregation_prompt(
        task, samples, best_so_far
    )

    assert "Original Task:" in prompt
    assert task in prompt
    assert "Current Best Response" in prompt
    assert best_so_far in prompt
    assert "Candidate Responses to Synthesize" in prompt
    assert "Sample 1" in prompt
    assert "Sample 2" in prompt
    assert "Sample 3" in prompt


def test_format_aggregation_prompt_no_best(basic_seq):
    """Test aggregation prompt formatting without best_so_far."""
    task = "Test task"
    samples = ["Sample 1", "Sample 2"]

    prompt = basic_seq._format_aggregation_prompt(task, samples)

    assert "Original Task:" in prompt
    assert task in prompt
    assert "Current Best Response" not in prompt
    assert "Candidate Responses to Synthesize" in prompt
    assert "Sample 1" in prompt
    assert "Sample 2" in prompt


def test_aggregate_window_success(basic_seq, mock_agents):
    """Test successful window aggregation."""
    _, aggregator = mock_agents
    aggregator.run.return_value = "Aggregated result"

    with patch.object(basic_seq, "aggregator", aggregator):
        result = basic_seq._aggregate_window(
            "Test task", ["Sample 1", "Sample 2"], "Best so far"
        )

        assert result == "Aggregated result"
        assert basic_seq.metrics["total_aggregations"] == 1


def test_aggregate_window_with_retry(basic_seq, mock_agents):
    """Test window aggregation with retry on failure."""
    _, aggregator = mock_agents
    aggregator.run.side_effect = [
        Exception("First failure"),
        Exception("Second failure"),
        "Aggregated result",
    ]

    with patch.object(basic_seq, "aggregator", aggregator):
        result = basic_seq._aggregate_window(
            "Test task", ["Sample 1", "Sample 2"]
        )

        assert result == "Aggregated result"
        assert basic_seq.metrics["total_aggregations"] == 1


def test_run_method_success(basic_seq, mock_agents):
    """Test successful run method execution."""
    proposer, aggregator = mock_agents

    # Configure mocks
    proposer.run.return_value = "Generated sample"
    aggregator.run.return_value = "Aggregated result"

    with patch.object(basic_seq, "proposer", proposer), patch.object(
        basic_seq, "aggregator", aggregator
    ):

        result = basic_seq.run("Test task")

        assert isinstance(result, dict)
        assert "final_output" in result
        assert "all_samples" in result
        assert "aggregation_steps" in result
        assert "metrics" in result
        assert "task" in result
        assert "timestamp" in result

        assert result["task"] == "Test task"
        assert len(result["all_samples"]) == 3
        assert result["final_output"] == "Aggregated result"


def test_run_method_with_retry(basic_seq, mock_agents):
    """Test run method with retry on failure."""
    proposer, aggregator = mock_agents

    # Configure mocks to fail first time, succeed second time
    proposer.run.side_effect = [
        Exception("First failure"),
        "Sample 1",
        "Sample 2",
        "Sample 3",
    ]

    aggregator.run.side_effect = [
        Exception("Aggregation failure"),
        "Final result",
    ]

    with patch.object(basic_seq, "proposer", proposer), patch.object(
        basic_seq, "aggregator", aggregator
    ):

        result = basic_seq.run("Test task")

        assert result is not None
        assert result["final_output"] == "Final result"


# ============================================================================
# Error Handling and Edge Cases Tests
# ============================================================================


def test_run_invalid_task(basic_seq):
    """Test run method with invalid task input."""
    with pytest.raises(
        ValueError, match="task must be a non-empty string"
    ):
        basic_seq.run("")

    with pytest.raises(
        ValueError, match="task must be a non-empty string"
    ):
        basic_seq.run(None)


def test_run_max_iterations_reached(basic_seq, mock_agents):
    """Test run method when max iterations are reached."""
    proposer, aggregator = mock_agents

    # Configure mocks
    proposer.run.return_value = "Generated sample"
    aggregator.run.return_value = "Aggregated result"

    # Set max_iterations to 1 to trigger the warning
    basic_seq.max_iterations = 1

    with patch.object(basic_seq, "proposer", proposer), patch.object(
        basic_seq, "aggregator", aggregator
    ):

        result = basic_seq.run("Test task")

        assert result is not None
        assert result["aggregation_steps"] <= 1


def test_generate_samples_exception_propagation(
    basic_seq, mock_agents
):
    """Test that exceptions in sample generation are properly propagated."""
    proposer, _ = mock_agents
    proposer.run.side_effect = Exception("Persistent failure")

    with patch.object(basic_seq, "proposer", proposer):
        with pytest.raises(Exception, match="Persistent failure"):
            basic_seq._generate_samples("Test task", 3)


def test_aggregate_window_exception_propagation(
    basic_seq, mock_agents
):
    """Test that exceptions in window aggregation are properly propagated."""
    _, aggregator = mock_agents
    aggregator.run.side_effect = Exception(
        "Persistent aggregation failure"
    )

    with patch.object(basic_seq, "aggregator", aggregator):
        with pytest.raises(
            Exception, match="Persistent aggregation failure"
        ):
            basic_seq._aggregate_window(
                "Test task", ["Sample 1", "Sample 2"]
            )


def test_run_exception_propagation(basic_seq, mock_agents):
    """Test that exceptions in run method are properly propagated."""
    proposer, _ = mock_agents
    proposer.run.side_effect = Exception("Persistent run failure")

    with patch.object(basic_seq, "proposer", proposer):
        with pytest.raises(Exception, match="Persistent run failure"):
            basic_seq.run("Test task")


# ============================================================================
# Metrics and Logging Tests
# ============================================================================


def test_metrics_initialization(basic_seq):
    """Test that metrics are properly initialized."""
    metrics = basic_seq.get_metrics()

    assert isinstance(metrics, dict)
    assert "total_samples_generated" in metrics
    assert "total_aggregations" in metrics
    assert "total_tokens_used" in metrics
    assert "execution_time_seconds" in metrics

    assert metrics["total_samples_generated"] == 0
    assert metrics["total_aggregations"] == 0
    assert metrics["total_tokens_used"] == 0
    assert metrics["execution_time_seconds"] == 0


def test_metrics_tracking(basic_seq, mock_agents):
    """Test that metrics are properly tracked during execution."""
    proposer, aggregator = mock_agents

    proposer.run.return_value = "Generated sample"
    aggregator.run.return_value = "Aggregated result"

    with patch.object(basic_seq, "proposer", proposer), patch.object(
        basic_seq, "aggregator", aggregator
    ):

        result = basic_seq.run("Test task")

        metrics = result["metrics"]
        assert metrics["total_samples_generated"] == 3
        assert metrics["total_aggregations"] >= 1
        assert metrics["execution_time_seconds"] > 0


def test_log_summary(basic_seq):
    """Test _log_summary method."""
    result = {
        "final_output": "Test output",
        "aggregation_steps": 2,
        "metrics": {
            "total_samples_generated": 3,
            "execution_time_seconds": 1.5,
        },
    }

    # This should not raise an exception
    basic_seq._log_summary(result)


def test_get_metrics_returns_copy(basic_seq):
    """Test that get_metrics returns a copy of metrics."""
    metrics1 = basic_seq.get_metrics()
    metrics2 = basic_seq.get_metrics()

    # Should be different objects
    assert metrics1 is not metrics2

    # But should have same content
    assert metrics1 == metrics2


# ============================================================================
# Integration Tests
# ============================================================================


def test_full_integration_small_samples():
    """Test full integration with small number of samples."""
    seq = SelfMoASeq(
        num_samples=2,
        window_size=3,
        reserved_slots=1,
        max_iterations=2,
        verbose=False,
        enable_logging=False,
    )

    proposer, aggregator = Mock(), Mock()
    proposer.run.return_value = "Generated sample"
    aggregator.run.return_value = "Aggregated result"

    with patch.object(seq, "proposer", proposer), patch.object(
        seq, "aggregator", aggregator
    ):

        result = seq.run("Integration test task")

        assert result is not None
        assert result["task"] == "Integration test task"
        assert len(result["all_samples"]) == 2
        assert result["final_output"] == "Aggregated result"
        assert result["aggregation_steps"] >= 0


def test_model_name_overrides():
    """Test that model name overrides work correctly."""
    seq = SelfMoASeq(
        model_name="base-model",
        proposer_model_name="proposer-model",
        aggregator_model_name="aggregator-model",
        verbose=False,
        enable_logging=False,
    )

    # The agents should be initialized with the override names
    assert seq.proposer.model_name == "proposer-model"
    assert seq.aggregator.model_name == "aggregator-model"


def test_temperature_settings():
    """Test that temperature settings are applied correctly."""
    seq = SelfMoASeq(
        temperature=0.5, verbose=False, enable_logging=False
    )

    assert seq.proposer.temperature == 0.5
    assert (
        seq.aggregator.temperature == 0.0
    )  # Deterministic aggregation


# ============================================================================
# Performance and Edge Case Tests
# ============================================================================


def test_minimum_valid_configuration():
    """Test with minimum valid configuration."""
    seq = SelfMoASeq(
        window_size=2,
        reserved_slots=1,
        max_iterations=1,
        num_samples=2,
        verbose=False,
        enable_logging=False,
    )

    assert seq.window_size == 2
    assert seq.reserved_slots == 1
    assert seq.max_iterations == 1
    assert seq.num_samples == 2


def test_zero_retries():
    """Test with zero retries (should still work but not retry)."""
    seq = SelfMoASeq(
        max_retries=0,
        num_samples=2,
        verbose=False,
        enable_logging=False,
    )

    assert seq.max_retries == 0

    proposer, aggregator = Mock(), Mock()
    proposer.run.return_value = "Generated sample"
    aggregator.run.return_value = "Aggregated result"

    with patch.object(seq, "proposer", proposer), patch.object(
        seq, "aggregator", aggregator
    ):

        result = seq.run("Test task")
        assert result is not None


def test_large_configuration():
    """Test with large configuration values."""
    seq = SelfMoASeq(
        window_size=20,
        reserved_slots=5,
        max_iterations=50,
        num_samples=100,
        max_retries=10,
        retry_delay=5.0,
        retry_backoff_multiplier=3.0,
        retry_max_delay=300.0,
        verbose=False,
        enable_logging=False,
    )

    assert seq.window_size == 20
    assert seq.reserved_slots == 5
    assert seq.max_iterations == 50
    assert seq.num_samples == 100
    assert seq.max_retries == 10
    assert seq.retry_delay == 5.0
    assert seq.retry_backoff_multiplier == 3.0
    assert seq.retry_max_delay == 300.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
