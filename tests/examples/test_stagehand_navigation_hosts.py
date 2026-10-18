"""Exercise actual navigation dispatch without a browser or providers."""

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "examples/tools/stagehand/1_stagehand_wrapper_agent.py"
)


@pytest.fixture
def navigate() -> Any:
    tree = ast.parse(EXAMPLE.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "StagehandAgent"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "_execute_browser_task"
    )
    namespace = {"Dict": dict, "Any": Any, "logger": Mock()}
    exec(  # noqa: S102 - Exercise checked-in code with offline dependencies.
        compile(
            ast.Module(body=[method], type_ignores=[]),
            str(EXAMPLE),
            "exec",
        ),
        namespace,
    )
    return namespace["_execute_browser_task"]


@pytest.mark.parametrize(
    ("task", "url"),
    [
        ("Open docs.python.org", "https://docs.python.org"),
        ("Visit my-example.com", "https://my-example.com"),
        ("Go to example.dev", "https://example.dev"),
        (
            "Open example.com/docs/start",
            "https://example.com/docs/start",
        ),
        (
            "Visit api.example.net:8080/v1",
            "https://api.example.net:8080/v1",
        ),
        (
            "Navigate to example.org?q=test",
            "https://example.org?q=test",
        ),
        ("Open example.com", "https://example.com"),
        (
            "Go to https://example.org/a?b=2",
            "https://example.org/a?b=2",
        ),
    ],
)
def test_complete_destination_reaches_browser(
    navigate: Any, task: str, url: str
) -> None:
    page = SimpleNamespace(
        goto=AsyncMock(), act=AsyncMock(return_value="ok")
    )
    agent = SimpleNamespace(stagehand=SimpleNamespace(page=page))
    result = asyncio.run(navigate(agent, task))
    page.goto.assert_awaited_once_with(url)
    assert result["data"]["navigated_to"] == url


def test_no_destination_does_not_navigate(navigate: Any) -> None:
    page = SimpleNamespace(
        goto=AsyncMock(), act=AsyncMock(return_value="ok")
    )
    agent = SimpleNamespace(stagehand=SimpleNamespace(page=page))
    result = asyncio.run(navigate(agent, "Open the settings panel"))
    page.goto.assert_not_awaited()
    assert "navigated_to" not in result["data"]
