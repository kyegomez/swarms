import subprocess
import sys
import pytest
from unittest.mock import patch
from swarms.utils.litellm_wrapper import LiteLLM, NetworkConnectionError


def test_import_swarms_does_not_load_requests():
    """Importing swarms in a fresh process must not pull in requests."""
    cmd = [
        sys.executable,
        "-c",
        "import sys, swarms; assert 'requests' not in sys.modules, f'requests imported into sys.modules: {sys.modules.get(\"requests\")}'",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"Subprocess failed with stderr:\n{proc.stderr}"


def test_network_errors_tuple_does_not_use_requests():
    """_NETWORK_ERRORS should contain standard connection/timeout exceptions without requests."""
    assert ConnectionError in LiteLLM._NETWORK_ERRORS
    assert TimeoutError in LiteLLM._NETWORK_ERRORS
    assert OSError in LiteLLM._NETWORK_ERRORS
    for cls in LiteLLM._NETWORK_ERRORS:
        module = getattr(cls, "__module__", "")
        assert "requests" not in module, f"Found requests exception class in _NETWORK_ERRORS: {cls}"


def test_network_connection_error_raised_on_connection_failure():
    """Builtin ConnectionError during run() must be caught and raised as NetworkConnectionError."""
    llm = LiteLLM(model_name="gpt-4o")
    with patch("swarms.utils.litellm_wrapper.completion", side_effect=ConnectionError("Failed to reach server")):
        with pytest.raises(NetworkConnectionError) as exc_info:
            llm.run("Hello")
        assert "Network error while connecting" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, ConnectionError)
