"""Exercise the callable tool through its real synchronous wrapper."""

import ast
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "examples/tools/browser_use/browser_use_as_tool.py"
)


@pytest.fixture
def tool() -> Any:
    definitions = [
        node
        for node in ast.parse(EXAMPLE.read_text()).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    ]
    result = SimpleNamespace(
        model_dump_json=Mock(
            return_value=json.dumps({"result": "ok"})
        )
    )
    browser = SimpleNamespace(run=AsyncMock(return_value=result))
    llm = Mock()

    def browser_factory(task: str, llm: Any = None) -> Any:
        browser.task = task
        browser.llm = llm
        return browser

    factory = Mock(side_effect=browser_factory)
    namespace = {
        "asyncio": asyncio,
        "BrowserAgent": factory,
        "ChatOpenAI": Mock(return_value=llm),
    }
    exec(  # noqa: S102 - Load real definitions without provider imports.
        compile(
            ast.Module(body=definitions, type_ignores=[]),
            str(EXAMPLE),
            "exec",
        ),
        namespace,
    )
    return SimpleNamespace(
        call=namespace["browser_agent_tool"],
        browser=browser,
        factory=factory,
        result=result,
        llm=llm,
    )


@pytest.mark.parametrize("task", ["Read example.org", "查找公开文档"])
def test_tool_returns_serialized_browser_result(
    tool: Any, task: str
) -> None:
    assert json.loads(tool.call(task)) == {"result": "ok"}
    tool.factory.assert_called_once_with(task=task, llm=tool.llm)
    tool.browser.run.assert_awaited_once_with()
    tool.result.model_dump_json.assert_called_once_with(indent=4)


def test_browser_failure_propagates(tool: Any) -> None:
    tool.browser.run.side_effect = RuntimeError(
        "Offline browser failure"
    )
    with pytest.raises(RuntimeError, match="Offline browser failure"):
        tool.call("Read example.org")
