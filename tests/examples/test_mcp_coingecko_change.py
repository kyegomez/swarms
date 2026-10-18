"""Keep CoinGecko prices usable when 24-hour change is unavailable."""

import runpy
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "examples/mcp/servers/crypto_price_server.py"
)


@pytest.fixture
def price_tool(monkeypatch: pytest.MonkeyPatch) -> Callable:
    server = SimpleNamespace(tool=lambda **kwargs: lambda fn: fn)
    monkeypatch.setitem(
        sys.modules,
        "mcp.server.fastmcp",
        SimpleNamespace(FastMCP=lambda name: server),
    )
    blocked = Mock(side_effect=AssertionError("No live requests"))
    monkeypatch.setattr("socket.socket", blocked)
    monkeypatch.setattr("socket.create_connection", blocked)
    return runpy.run_path(str(EXAMPLE))["get_crypto_price"]


@pytest.mark.parametrize(
    ("fields", "expected_change"),
    [
        ({"usd_24h_change": None}, "N/A"),
        ({}, "N/A"),
        ({"usd_24h_change": 0}, "0.00%"),
        ({"usd_24h_change": -1.25}, "-1.25%"),
        ({"usd_24h_change": 2.5}, "2.50%"),
    ],
)
def test_available_price_survives_optional_change(
    monkeypatch: pytest.MonkeyPatch,
    price_tool: Callable,
    fields: dict,
    expected_change: str,
) -> None:
    response = Mock()
    response.json.return_value = {
        "bitcoin": {"usd": 12345.67, **fields}
    }
    get = Mock(return_value=response)
    monkeypatch.setattr(requests, "get", get)

    assert price_tool("bitcoin") == (
        "Current price of Bitcoin: $12,345.67\n"
        f"24h Change: {expected_change}"
    )
    response.raise_for_status.assert_called_once_with()
    assert "include_24hr_change=true" in get.call_args.args[0]


def test_unknown_coin_keeps_existing_message(
    monkeypatch: pytest.MonkeyPatch, price_tool: Callable
) -> None:
    response = Mock()
    response.json.return_value = {}
    monkeypatch.setattr(requests, "get", Mock(return_value=response))
    assert price_tool("unknown") == (
        "Could not find data for unknown. "
        "Please check the cryptocurrency ID."
    )


def test_http_error_keeps_existing_message(
    monkeypatch: pytest.MonkeyPatch, price_tool: Callable
) -> None:
    response = Mock()
    response.raise_for_status.side_effect = requests.HTTPError(
        "offline HTTP failure"
    )
    monkeypatch.setattr(requests, "get", Mock(return_value=response))
    assert price_tool("bitcoin") == (
        "Error fetching crypto data: offline HTTP failure"
    )
