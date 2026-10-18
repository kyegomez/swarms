"""Verify prediction status codes without API requests."""

import ast
import json
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Union
from unittest.mock import Mock

import pytest
from requests import Response

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "examples/single_agent/capabilities/tools/omni_modal_agent.py"
)


@pytest.fixture
def tool() -> Any:
    definitions = [
        node
        for node in ast.parse(SOURCE.read_text()).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    ]
    response = Response()
    response._content = json.dumps(
        {"id": "offline", "status": "starting"}
    ).encode()
    transport = Mock(return_value=response)
    namespace = {
        "Enum": Enum,
        "Union": Union,
        "Dict": dict,
        "Any": Any,
        "os": SimpleNamespace(
            getenv=Mock(return_value="offline-fixture")
        ),
        "requests": SimpleNamespace(post=transport),
    }
    exec(  # noqa: S102 - Exclude the module's paid demo calls.
        compile(
            ast.Module(body=definitions, type_ignores=[]),
            str(SOURCE),
            "exec",
        ),
        namespace,
    )
    return SimpleNamespace(
        run=namespace["generate_content"],
        response=response,
        post=transport,
    )


@pytest.mark.parametrize("modality", ["image", "video", "music"])
@pytest.mark.parametrize("status", [200, 201, 202])
def test_successful_creation_returns_prediction(
    tool: Any, modality: str, status: int
) -> None:
    tool.response.status_code = status
    assert tool.run(modality, "Offline prompt") == {
        "id": "offline",
        "status": "starting",
    }
    tool.post.assert_called_once()
    assert (
        tool.post.call_args.kwargs["json"]["input"]["prompt"]
        == "Offline prompt"
    )


@pytest.mark.parametrize("status", [400, 401, 429, 500])
def test_http_failure_keeps_existing_error_contract(
    tool: Any, status: int
) -> None:
    tool.response.status_code = status
    with pytest.raises(RuntimeError, match=f"status {status}"):
        tool.run("image", "Offline prompt")


def test_unsupported_modality_does_not_post(tool: Any) -> None:
    with pytest.raises(ValueError, match="Unsupported modality"):
        tool.run("unknown", "Offline prompt")
    tool.post.assert_not_called()
