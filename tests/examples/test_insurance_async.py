import asyncio
import runpy
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, Mock

import pytest


@pytest.fixture
def example(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Load the real bot with separate synchronous/asynchronous APIs."""
    swarms = ModuleType("swarms")
    swarms.Agent = Mock()
    swarms.Agent.return_value.run = Mock(return_value="sync answer")
    swarms.Agent.return_value.arun = AsyncMock(
        return_value="async answer"
    )
    monkeypatch.setitem(sys.modules, "swarms", swarms)
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/guides/demos/insurance/insurance_swarm.py"
    )
    return runpy.run_path(str(path))


@pytest.mark.parametrize(
    ("node", "user_input"),
    [("MAIN_MENU", choice) for choice in "123456x"]
    + [
        ("LOCATE_OFFICE", "Quito"),
        ("QUOTE_REQUEST", "AUTO001"),
        ("QUOTE_REQUEST", "Help"),
    ],
)
def test_bot_awaits_async_agent(
    example: dict, node: str, user_input: str
) -> None:
    """Return a response through every menu and follow-up path."""
    bot = example["InsuranceBot"]()
    bot.current_node = example["WorkflowNode"][node]

    assert asyncio.run(bot.process_user_input(user_input)) == (
        "async answer"
    )
    bot.agent.arun.assert_awaited_once()
    bot.agent.run.assert_not_called()


def test_error_recovery_also_awaits_async_agent(
    example: dict,
) -> None:
    """Reset the menu and await the fallback after an agent error."""
    bot = example["InsuranceBot"]()
    bot.current_node = example["WorkflowNode"].LOCATE_OFFICE
    bot.agent.arun.side_effect = [RuntimeError("offline"), "help"]

    assert asyncio.run(bot.process_user_input("Quito")) == "help"
    assert bot.current_node == example["WorkflowNode"].MAIN_MENU
    assert bot.agent.arun.await_count == 2
    bot.agent.run.assert_not_called()
