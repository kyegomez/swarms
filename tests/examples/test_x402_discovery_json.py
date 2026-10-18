"""Offline regression tests for the x402 discovery tool's JSON output."""

import json
import runpy
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest


@pytest.fixture
def discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Callable, Mock]:
    """Load the real example with Agent and HTTP client test doubles."""
    swarms = ModuleType("swarms")
    swarms.Agent = Mock()
    httpx = ModuleType("httpx")
    client = Mock(get=AsyncMock())
    client.get.return_value = Mock()
    context = MagicMock()
    context.__aenter__.return_value = client
    httpx.AsyncClient = Mock(return_value=context)
    monkeypatch.setitem(sys.modules, "swarms", swarms)
    monkeypatch.setitem(sys.modules, "httpx", httpx)
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/guides/x402_examples/agent_integration"
        / "x402_discovery_query.py"
    )
    example = runpy.run_path(str(path))
    return example["get_x402_services_sync"], client


@pytest.mark.parametrize(
    "services",
    [
        [],
        [
            {
                "resource": "https://example.com",
                "name": "Reader's service",
            }
        ],
        [
            {
                "metadata": {
                    "enabled": True,
                    "description": None,
                    "name": "查询",
                }
            }
        ],
    ],
)
def test_discovery_tool_returns_parseable_json(
    discovery: tuple[Callable, Mock], services: list[dict]
) -> None:
    """Round-trip real wrapper output, including null and Boolean metadata."""
    get_services, client = discovery
    client.get.return_value.json.return_value = {"items": services}

    result = get_services(limit=2, offset=3)

    assert json.loads(result) == services
    client.get.assert_awaited_once_with(
        "https://api.cdp.coinbase.com/platform/v2/x402/discovery/resources",
        params={"offset": 3, "limit": 2},
    )
    client.get.return_value.raise_for_status.assert_called_once()
