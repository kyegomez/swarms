import pytest
from swarms.structs.council_as_judge import (
    EvaluationError,
    DimensionEvaluationError,
    AggregationError,
    EVAL_DIMENSIONS,
    judge_system_prompt,
    build_judge_prompt,
    aggregator_system_prompt,
    build_aggregation_prompt,
)


def test_evaluation_error_is_exception():
    """Test that EvaluationError is an Exception subclass"""
    assert issubclass(EvaluationError, Exception)


def test_dimension_evaluation_error_is_evaluation_error():
    """Test that DimensionEvaluationError is an EvaluationError subclass"""
    assert issubclass(DimensionEvaluationError, EvaluationError)


def test_aggregation_error_is_evaluation_error():
    """Test that AggregationError is an EvaluationError subclass"""
    assert issubclass(AggregationError, EvaluationError)


def test_eval_dimensions_exists():
    """Test that EVAL_DIMENSIONS dictionary exists"""
    assert isinstance(EVAL_DIMENSIONS, dict)
    assert len(EVAL_DIMENSIONS) > 0


def test_eval_dimensions_contains_expected_keys():
    """Test that EVAL_DIMENSIONS contains expected evaluation dimensions"""
    expected_dimensions = [
        "accuracy",
        "helpfulness",
        "harmlessness",
        "coherence",
        "conciseness",
        "instruction_adherence",
    ]
    for dimension in expected_dimensions:
        assert dimension in EVAL_DIMENSIONS


def test_eval_dimensions_values_are_strings():
    """Test that all EVAL_DIMENSIONS values are strings"""
    for dimension, description in EVAL_DIMENSIONS.items():
        assert isinstance(description, str)
        assert len(description) > 0


def test_judge_system_prompt_returns_string():
    """Test that judge_system_prompt returns a string"""
    result = judge_system_prompt()
    assert isinstance(result, str)
    assert len(result) > 0


def test_judge_system_prompt_contains_key_phrases():
    """Test that judge_system_prompt contains expected content"""
    result = judge_system_prompt()
    assert "evaluator" in result.lower()
    assert "feedback" in result.lower()


def test_build_judge_prompt_valid_dimension():
    """Test build_judge_prompt with valid dimension"""
    result = build_judge_prompt(
        dimension_name="accuracy",
        task="Test task",
        task_response="Test response"
    )
    assert isinstance(result, str)
    assert "accuracy" in result.lower()
    assert "Test task" in result
    assert "Test response" in result


def test_build_judge_prompt_invalid_dimension_raises_error():
    """Test that build_judge_prompt raises KeyError for invalid dimension"""
    with pytest.raises(KeyError, match="Unknown evaluation dimension"):
        build_judge_prompt(
            dimension_name="invalid_dimension",
            task="Test task",
            task_response="Test response"
        )


def test_build_judge_prompt_includes_evaluation_focus():
    """Test that build_judge_prompt includes evaluation focus from EVAL_DIMENSIONS"""
    result = build_judge_prompt(
        dimension_name="helpfulness",
        task="Test task",
        task_response="Test response"
    )
    # Should contain some content from EVAL_DIMENSIONS["helpfulness"]
    assert "helpfulness" in result.lower()


def test_aggregator_system_prompt_returns_string():
    """Test that aggregator_system_prompt returns a string"""
    result = aggregator_system_prompt()
    assert isinstance(result, str)
    assert len(result) > 0


def test_aggregator_system_prompt_contains_key_phrases():
    """Test that aggregator_system_prompt contains expected content"""
    result = aggregator_system_prompt()
    assert "synthesizing" in result.lower() or "synthesis" in result.lower()
    assert "report" in result.lower()


def test_build_aggregation_prompt_basic():
    """Test build_aggregation_prompt with basic input"""
    rationales = {
        "accuracy": "This response is accurate",
        "helpfulness": "This response is helpful"
    }
    result = build_aggregation_prompt(rationales)
    assert isinstance(result, str)
    assert "accuracy" in result.lower()
    assert "helpfulness" in result.lower()
    assert "This response is accurate" in result
    assert "This response is helpful" in result


def test_build_aggregation_prompt_empty_dict():
    """Test build_aggregation_prompt with empty dictionary"""
    result = build_aggregation_prompt({})
    assert isinstance(result, str)
    assert len(result) > 0


def test_build_aggregation_prompt_single_dimension():
    """Test build_aggregation_prompt with single dimension"""
    rationales = {"accuracy": "Accuracy analysis"}
    result = build_aggregation_prompt(rationales)
    assert "accuracy" in result.lower()
    assert "Accuracy analysis" in result


def test_build_aggregation_prompt_multiple_dimensions():
    """Test build_aggregation_prompt with multiple dimensions"""
    rationales = {
        "accuracy": "Accuracy text",
        "helpfulness": "Helpfulness text",
        "coherence": "Coherence text"
    }
    result = build_aggregation_prompt(rationales)
    for dim, text in rationales.items():
        assert dim.upper() in result
        assert text in result


def test_evaluation_error_can_be_raised():
    """Test that EvaluationError can be raised"""
    with pytest.raises(EvaluationError, match="Test error"):
        raise EvaluationError("Test error")


def test_dimension_evaluation_error_can_be_raised():
    """Test that DimensionEvaluationError can be raised"""
    with pytest.raises(DimensionEvaluationError, match="Dimension error"):
        raise DimensionEvaluationError("Dimension error")


def test_aggregation_error_can_be_raised():
    """Test that AggregationError can be raised"""
    with pytest.raises(AggregationError, match="Aggregation error"):
        raise AggregationError("Aggregation error")


def test_judge_system_prompt_is_cacheable():
    """Test that judge_system_prompt can be called multiple times"""
    result1 = judge_system_prompt()
    result2 = judge_system_prompt()
    assert result1 == result2


def test_aggregator_system_prompt_is_cacheable():
    """Test that aggregator_system_prompt can be called multiple times"""
    result1 = aggregator_system_prompt()
    result2 = aggregator_system_prompt()
    assert result1 == result2
