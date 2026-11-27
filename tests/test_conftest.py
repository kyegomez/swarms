import importlib.util
from pathlib import Path
import pytest
from unittest.mock import MagicMock

conftest_path = Path(__file__).parent / "conftest.py"
spec = importlib.util.spec_from_file_location(
    "conftest_module", conftest_path
)
conftest_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conftest_module)

_is_credential_failure = conftest_module._is_credential_failure
_extract_all_error_messages = (
    conftest_module._extract_all_error_messages
)
CREDENTIAL_ERROR_PATTERNS = conftest_module.CREDENTIAL_ERROR_PATTERNS


def test_is_credential_failure_matches_known_patterns():
    """Verify that _is_credential_failure detects credential error strings."""
    assert _is_credential_failure(
        Exception(
            "litellm.InternalServerError: OpenAIException - Missing credentials."
        )
    )
    assert _is_credential_failure(
        RuntimeError("Please set OPENAI_API_KEY in your environment.")
    )
    assert _is_credential_failure(
        ValueError("status_code: 401 unauthorized")
    )
    assert _is_credential_failure(
        Exception("AuthenticationError: Incorrect API key provided")
    )
    assert _is_credential_failure(
        Exception("No API key provided for ANTHROPIC_API_KEY")
    )


def test_is_credential_failure_ignores_unrelated_errors():
    """Verify that _is_credential_failure does not match regular errors."""
    assert not _is_credential_failure(
        ValueError("Invalid argument: foo")
    )
    assert not _is_credential_failure(KeyError("missing_key"))
    assert not _is_credential_failure(
        AssertionError("expected 1 == 2")
    )
    assert not _is_credential_failure(
        TypeError("unsupported operand type")
    )
    assert not _is_credential_failure(
        ZeroDivisionError("division by zero")
    )


def test_credential_patterns_tuple_is_non_empty():
    """Verify that CREDENTIAL_ERROR_PATTERNS has common provider keys."""
    assert "openai_api_key" in CREDENTIAL_ERROR_PATTERNS
    assert "missing credentials" in CREDENTIAL_ERROR_PATTERNS
    assert "unauthorized" in CREDENTIAL_ERROR_PATTERNS


def test_is_credential_failure_unwraps_chained_causes():
    """Verify that _is_credential_failure detects errors hidden in __cause__ or __context__."""
    root_cause = Exception(
        "Missing credentials. Please pass an api_key"
    )
    wrapper = RuntimeError("Execution failed")
    wrapper.__cause__ = root_cause
    assert _is_credential_failure(wrapper)

    context_err = ValueError("Invalid API key")
    wrapper2 = RuntimeError("Wrapper failed")
    wrapper2.__context__ = context_err
    assert _is_credential_failure(wrapper2)


def test_is_credential_failure_unwraps_retry_error():
    """Verify that _is_credential_failure extracts exceptions from last_attempt."""
    mock_retry = MagicMock(spec=Exception)
    mock_attempt = MagicMock()
    mock_attempt.exception.return_value = Exception(
        "OpenAIException: did not find an openai_api_key"
    )
    mock_retry.last_attempt = mock_attempt
    mock_retry.__cause__ = None
    mock_retry.__context__ = None
    assert _is_credential_failure(mock_retry)


def test_extract_all_error_messages_handles_circular_references():
    """Verify that _extract_all_error_messages does not loop indefinitely on cyclic exception chains."""
    err1 = RuntimeError("error 1")
    err2 = RuntimeError("error 2")
    err1.__cause__ = err2
    err2.__cause__ = err1
    msg = _extract_all_error_messages(err1)
    assert "error 1" in msg
    assert "error 2" in msg
