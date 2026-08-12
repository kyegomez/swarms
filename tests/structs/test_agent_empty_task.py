"""Regression tests for empty-task handling in Agent.run().

Covers the bug where a commented-out `self.interactive and ...` guard caused
`run()` to block on stdin for an empty or None task even when
`interactive=False`. The non-interactive path must fail fast with a clear
error and never read from the console.
"""

import pytest

from swarms.structs.agent import Agent
from swarms.utils.formatter import formatter


@pytest.mark.parametrize("empty_task", ["", "   ", "\n\t ", None])
def test_empty_task_non_interactive_raises_value_error(empty_task):
    agent = Agent(agent_name="NoTaskAgent", interactive=False, max_loops=1)
    with pytest.raises(ValueError, match="No task provided"):
        agent.run(empty_task)


def test_empty_task_non_interactive_does_not_read_stdin(monkeypatch):
    """The non-interactive empty-task path must not touch console input."""
    called = {"input": False}

    def _fail_input(*args, **kwargs):
        called["input"] = True
        raise AssertionError(
            "console.input() was called with interactive=False"
        )

    monkeypatch.setattr(formatter.console, "input", _fail_input)

    agent = Agent(agent_name="NoStdinAgent", interactive=False, max_loops=1)
    with pytest.raises(ValueError, match="No task provided"):
        agent.run("")

    assert called["input"] is False
