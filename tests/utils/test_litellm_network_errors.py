"""``import swarms`` does not load requests, and network failures from a
model call still surface as NetworkConnectionError (#2931)."""

import subprocess
import sys

import pytest

from swarms.utils import litellm_wrapper, llm_backend
from swarms.utils.litellm_wrapper import (
    LiteLLM,
    NetworkConnectionError,
)


def test_import_swarms_does_not_import_requests():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, swarms; print('requests' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.stdout.strip().endswith("False"), result.stderr


def _backend_connection_error():
    error_type = llm_backend.APIConnectionError
    try:
        return error_type(
            message="connection refused",
            llm_provider="openai",
            model="gpt-4o-mini",
        )
    except TypeError:
        return error_type("connection refused")


@pytest.mark.parametrize(
    "make_error",
    [
        lambda: ConnectionError("connection refused"),
        lambda: TimeoutError("timed out"),
        _backend_connection_error,
    ],
    ids=[
        "ConnectionError",
        "TimeoutError",
        "backend APIConnectionError",
    ],
)
def test_network_errors_become_network_connection_error(
    monkeypatch, make_error
):
    error = make_error()

    def _completion(**kwargs):
        raise error

    monkeypatch.setattr(litellm_wrapper, "completion", _completion)
    llm = LiteLLM(model_name="gpt-4o-mini")

    with pytest.raises(NetworkConnectionError) as excinfo:
        llm.run("hello")

    assert excinfo.value.__cause__ is error
