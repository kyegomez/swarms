"""Measure executed work rather than coroutine creation."""

import ast
import asyncio
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "examples/single_agent/capabilities/tools"
    / "example_async_vs_multithread.py"
)


@pytest.fixture
def timing():
    tree = ast.parse(SOURCE.read_text())
    decorator = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "measure_time_and_memory"
    )
    clock = SimpleNamespace(now=100.0)
    process = Mock(
        return_value=SimpleNamespace(
            memory_info=lambda: SimpleNamespace(rss=2 * 1024**2)
        )
    )
    namespace = {
        "time": SimpleNamespace(time=lambda: clock.now),
        "psutil": SimpleNamespace(Process=process),
        "inspect": inspect,
    }
    exec(  # noqa: S102 - Only the local timing decorator is loaded.
        compile(
            ast.Module(body=[decorator], type_ignores=[]),
            str(SOURCE),
            "exec",
        ),
        namespace,
    )
    return namespace["measure_time_and_memory"], clock, process


def test_async_elapsed_includes_awaited_work(timing, capsys):
    decorate, clock, process = timing

    @decorate
    async def job():
        await asyncio.sleep(0)
        clock.now += 5
        return "completed"

    assert asyncio.run(job()) == "completed"
    assert "Time: 5.00s | Memory: 2.00 MB" in capsys.readouterr().out
    process.assert_called_once()


def test_creating_coroutine_does_not_report_work(timing, capsys):
    decorate, _clock, process = timing

    @decorate
    async def job():
        return "completed"

    coroutine = job()
    try:
        assert capsys.readouterr().out == ""
        process.assert_not_called()
    finally:
        coroutine.close()


def test_sync_timing_and_return_value_are_preserved(timing, capsys):
    decorate, clock, process = timing

    @decorate
    def job():
        clock.now += 3
        return 42

    assert job() == 42
    assert "Time: 3.00s | Memory: 2.00 MB" in capsys.readouterr().out
    process.assert_called_once()


def test_failed_async_work_does_not_report_success(timing, capsys):
    decorate, _clock, process = timing

    @decorate
    async def job():
        raise RuntimeError("offline failure")

    with pytest.raises(RuntimeError, match="offline failure"):
        asyncio.run(job())
    assert capsys.readouterr().out == ""
    process.assert_not_called()
