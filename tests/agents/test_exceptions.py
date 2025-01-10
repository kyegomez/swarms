from swarms.agents.exceptions import (
    ErrorSeverity,
    ToolAgentError,
    ValidationError,
    ModelNotProvidedError
)

def test_error_severity():
    assert ErrorSeverity.LOW.value == "low"
    assert ErrorSeverity.MEDIUM.value == "medium"
    assert ErrorSeverity.HIGH.value == "high"
    assert ErrorSeverity.CRITICAL.value == "critical"

def test_tool_agent_error():
    error = ToolAgentError(
        "Test error",
        severity=ErrorSeverity.HIGH,
        details={"test": "value"}
    )
    assert str(error) == "Test error"
    assert error.severity == ErrorSeverity.HIGH
    assert error.details == {"test": "value"}

def test_validation_error():
    error = ValidationError("Validation failed")
    assert isinstance(error, ToolAgentError)
    assert str(error) == "Validation failed"

def test_model_not_provided_error():
    error = ModelNotProvidedError("Model missing")
    assert isinstance(error, ToolAgentError)
    assert str(error) == "Model missing"
