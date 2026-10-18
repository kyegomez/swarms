"""Offline regression tests for default Calendar event end times."""

import runpy
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, Mock

import pytest


@pytest.fixture
def calendar(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Callable, Mock]:
    """Load the real example with HTTP, Agent and speech clients stubbed."""
    modules = {
        name: ModuleType(name)
        for name in ("httpx", "swarms", "voice_agents")
    }
    client = Mock()
    client.post.return_value.json.return_value = {
        "id": "offline-event"
    }
    context = MagicMock()
    context.__enter__.return_value = client
    modules["httpx"].Client = Mock(return_value=context)
    modules["swarms"].Agent = Mock()
    modules["voice_agents"].StreamingTTSCallback = Mock()
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/guides/voice_agents/voice_agents_examples"
        / "google_calendar_agent.py"
    )
    example = runpy.run_path(str(path))
    return example["create_calendar_event"], client


@pytest.mark.parametrize(
    ("start", "expected_end"),
    [
        ("2026-01-15T10:30:00Z", "2026-01-15T11:30:00Z"),
        ("2026-01-15T23:30:00Z", "2026-01-16T00:30:00Z"),
        ("2026-01-31T23:30:00Z", "2026-02-01T00:30:00Z"),
        ("2026-12-31T23:30:00Z", "2027-01-01T00:30:00Z"),
        ("2024-02-28T23:30:00Z", "2024-02-29T00:30:00Z"),
        ("2026-01-15T23:30:00+05:30", "2026-01-16T00:30:00+05:30"),
    ],
)
def test_default_end_is_one_hour_later(
    calendar: tuple[Callable, Mock], start: str, expected_end: str
) -> None:
    """Default duration crosses calendar boundaries and preserves offsets."""
    create_event, client = calendar

    result = create_event(
        "Offline test", start_datetime=start, access_token="test-only"
    )

    assert result == {"id": "offline-event"}
    event = client.post.call_args.kwargs["json"]
    assert event["start"]["dateTime"] == start
    assert event["end"]["dateTime"] == expected_end
    client.post.assert_called_once()


def test_explicit_end_is_unchanged(
    calendar: tuple[Callable, Mock],
) -> None:
    """An explicitly supplied end time keeps its original duration."""
    create_event, client = calendar
    end = "2027-01-01T02:45:00Z"

    create_event(
        "Offline test",
        start_datetime="2026-12-31T23:30:00Z",
        end_datetime=end,
        access_token="test-only",
    )

    assert (
        client.post.call_args.kwargs["json"]["end"]["dateTime"] == end
    )
