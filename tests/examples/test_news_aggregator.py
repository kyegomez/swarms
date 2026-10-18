"""Offline regressions for the Hacker News example's API responses."""

import runpy
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, call

import pytest


@pytest.fixture
def example(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Load the actual example without creating an Agent or making requests."""
    modules = {
        name: ModuleType(name)
        for name in (
            "httpx",
            "swarms",
            "swarms.prompts",
            "swarms.prompts.summaries_prompts",
        )
    }
    modules["httpx"].get = Mock()
    modules["swarms"].Agent = Mock()
    modules["swarms.prompts.summaries_prompts"].SUMMARIZE_PROMPT = (
        "Summarize"
    )
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/guides/demos/news_aggregator_summarizer.py"
    )
    return runpy.run_path(str(path))


@pytest.mark.parametrize("payload", [None, {}, "unavailable"])
def test_invalid_topstories_returns_no_headlines(
    example: dict, payload: object
) -> None:
    """Missing or malformed top-story lists must not crash the example."""
    get = example["httpx"].get
    get.return_value.json.return_value = payload

    assert example["fetch_hackernews_headlines"]() == []
    get.assert_called_once()


@pytest.mark.parametrize("payload", [None, [], "unavailable"])
def test_invalid_item_does_not_discard_later_headlines(
    example: dict, payload: object
) -> None:
    """An unavailable item must not stop retrieval of subsequent stories."""
    story = {
        "title": "Available story",
        "url": "https://example.com/news",
    }
    get = example["httpx"].get
    get.side_effect = [
        Mock(json=Mock(return_value=value))
        for value in ([1, 2], payload, story)
    ]

    assert example["fetch_hackernews_headlines"]() == [story]
    assert get.call_count == 3


def test_valid_items_preserve_order_and_limit(example: dict) -> None:
    """Valid stories keep their API order and respect the requested limit."""
    stories = [
        {"title": "First", "url": "https://example.com/first"},
        {"title": "Second", "url": "https://example.com/second"},
    ]
    get = example["httpx"].get
    get.side_effect = [
        Mock(json=Mock(return_value=value))
        for value in ([1, 2, 3], *stories)
    ]

    assert example["fetch_hackernews_headlines"](2) == stories
    base = "https://hacker-news.firebaseio.com/v0"
    assert get.call_args_list == [
        call(f"{base}/topstories.json", timeout=10),
        call(f"{base}/item/1.json", timeout=10),
        call(f"{base}/item/2.json", timeout=10),
    ]
