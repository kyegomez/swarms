"""Exercise demo modes with the real token queues and offline model work."""

import ast
import asyncio
import builtins
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = (
    ROOT
    / "examples/single_agent/capabilities/streaming"
    / "example_streaming_tools.py"
)


@pytest.fixture
def offline_agent(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    tree = ast.parse((ROOT / "swarms/structs/agent.py").read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Agent"
    )
    cls.body = [
        node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in {"run_stream", "arun_stream"}
    ]
    namespace = {"threading": threading}
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            cls,
        ],
        type_ignores=[],
    )
    exec(  # noqa: S102 - Load the checked-in queue implementations.
        compile(
            ast.fix_missing_locations(module), "agent.py", "exec"
        ),
        namespace,
    )
    first_printed = threading.Event()
    tokens = []
    real_print = builtins.print

    def capture_print(*args, **kwargs):
        if kwargs.get("end") == "" and args:
            tokens.append(args[0])
            if args[0] == "first ":
                first_printed.set()
        real_print(*args, **kwargs)

    class OfflineAgent(namespace["Agent"]):
        def __init__(self, **kwargs):
            self.streaming_on = kwargs.get("streaming_on", False)

        def run(self, task, img=None, streaming_callback=None):
            if streaming_callback is not None:
                streaming_callback("first ")
                assert first_printed.wait(
                    2
                ), "First token not printed"
                streaming_callback("second")
            return "complete blocking answer"

    monkeypatch.setitem(
        sys.modules, "swarms", SimpleNamespace(Agent=OfflineAgent)
    )
    monkeypatch.setattr("builtins.print", capture_print)
    blocked = Mock(side_effect=AssertionError("Network disabled"))
    monkeypatch.setattr("socket.create_connection", blocked)
    monkeypatch.setattr("socket.socket.connect", blocked)
    monkeypatch.setattr("socket.socket.connect_ex", blocked)
    return tokens


@pytest.mark.parametrize("mode", ["async", "sync", "blocking"])
def test_selected_demo_mode(
    mode: str, offline_agent: list[str], capsys: pytest.CaptureFixture
) -> None:
    source = EXAMPLE.read_text().replace(
        'mode = "async"', f"mode = {mode!r}"
    )
    exec(  # noqa: S102 - Emulate editing the example's mode selector.
        compile(source, str(EXAMPLE), "exec"),
        {"__name__": "__main__"},
    )
    output = capsys.readouterr().out
    if mode == "blocking":
        assert "BLOCKING  (run)" in output
        assert "complete blocking answer" in output
        assert offline_agent == []
    else:
        expected = (
            "ASYNC STREAMING  (arun_stream)"
            if mode == "async"
            else "SYNC STREAMING  (run_stream)"
        )
        assert expected in output
        assert offline_agent == ["first ", "second"]
        assert "complete blocking answer" not in output


def test_direct_async_demo_streams(
    offline_agent: list[str], capsys: pytest.CaptureFixture
) -> None:
    namespace = {"__name__": "offline_example"}
    exec(  # noqa: S102 - Exercise the existing async function as a control.
        compile(EXAMPLE.read_text(), str(EXAMPLE), "exec"), namespace
    )
    asyncio.run(namespace["demo_async"]())
    assert offline_agent == ["first ", "second"]
    assert "first second" in capsys.readouterr().out
