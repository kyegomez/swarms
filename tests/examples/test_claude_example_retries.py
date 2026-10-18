"""Exercise retry settings with real Tenacity and an offline SDK."""

import asyncio
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from tenacity import RetryError

EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "examples/tools/claude/claude_as_a_tool.py"
)


@pytest.fixture
def example(monkeypatch):
    calls = []
    failures = SimpleNamespace(remaining=0)

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def query(self, task):
            calls.append(task)
            if failures.remaining:
                failures.remaining -= 1
                raise RuntimeError("offline SDK failure")

        async def receive_response(self):
            yield SimpleNamespace(
                content=[SimpleNamespace(text="completed")]
            )

    monkeypatch.setitem(
        sys.modules,
        "claude_code_sdk",
        SimpleNamespace(
            ClaudeCodeOptions=lambda **kwargs: kwargs,
            ClaudeSDKClient=Client,
        ),
    )
    monkeypatch.setitem(
        sys.modules, "dotenv", SimpleNamespace(load_dotenv=Mock())
    )
    monkeypatch.setitem(
        sys.modules, "loguru", SimpleNamespace(logger=Mock())
    )
    sleeper = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleeper)
    blocked = Mock(side_effect=AssertionError("No network access"))
    monkeypatch.setattr("socket.create_connection", blocked)
    monkeypatch.setattr("socket.socket.connect", blocked)
    namespace = runpy.run_path(str(EXAMPLE))
    return SimpleNamespace(
        cls=namespace["ClaudeAppGenerator"],
        calls=calls,
        failures=failures,
        sleep=sleeper,
    )


@pytest.mark.parametrize("attempts", [1, 2, 5])
def test_configured_attempt_limit_is_used(example, attempts):
    example.failures.remaining = 10
    generator = example.cls(retries=attempts, retry_delay=0)
    with pytest.raises(RetryError):
        asyncio.run(generator.generate_app_with_claude("offline"))
    assert len(example.calls) == attempts


def test_larger_budget_allows_recovery(example):
    example.failures.remaining = 4
    generator = example.cls(retries=5, retry_delay=0)
    assert generator.run("offline") == ["completed"]
    assert len(example.calls) == 5


def test_configured_delay_controls_backoff(example):
    example.failures.remaining = 2
    generator = example.cls(retries=3, retry_delay=0.25)
    assert generator.run("offline") == ["completed"]
    assert [
        call.args[0] for call in example.sleep.call_args_list
    ] == [
        0.25,
        0.5,
    ]


def test_first_attempt_success_does_not_wait(example):
    generator = example.cls(retries=5, retry_delay=0.25)
    assert generator.run("offline") == ["completed"]
    assert example.calls == ["offline"]
    example.sleep.assert_not_called()
