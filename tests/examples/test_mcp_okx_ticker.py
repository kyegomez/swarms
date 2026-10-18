"""Verify OKX ticker fields and spot-volume units offline."""

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "examples/mcp/servers/okx_crypto_server.py"
)


@pytest.fixture
def tools(monkeypatch: pytest.MonkeyPatch) -> dict:
    server = SimpleNamespace(
        tool=lambda **kwargs: lambda fn: fn,
        settings=SimpleNamespace(),
    )
    monkeypatch.setitem(
        sys.modules,
        "mcp.server.fastmcp",
        SimpleNamespace(FastMCP=lambda name: server),
    )
    blocked = Mock(side_effect=AssertionError("No live requests"))
    monkeypatch.setattr("socket.socket", blocked)
    monkeypatch.setattr("socket.create_connection", blocked)
    return runpy.run_path(str(EXAMPLE))


def mock_ticker(
    monkeypatch: pytest.MonkeyPatch, ticker: dict
) -> Mock:
    response = Mock()
    response.json.return_value = {
        "code": "0",
        "msg": "",
        "data": [ticker],
    }
    get = Mock(return_value=response)
    monkeypatch.setattr(requests, "get", get)
    return get


@pytest.mark.parametrize(
    ("last", "opening", "change"),
    [
        ("110", "100", "10.00%"),
        ("80", "100", "-20.00%"),
        ("100", "100", "0.00%"),
    ],
)
def test_change_uses_documented_opening_price(
    monkeypatch: pytest.MonkeyPatch,
    tools: dict,
    last: str,
    opening: str,
    change: str,
) -> None:
    get = mock_ticker(monkeypatch, {"last": last, "open24h": opening})
    output = tools["get_okx_crypto_price"]("btc")
    assert output == (
        f"Current price of BTC/USDT: ${float(last):,.2f}\n"
        f"24h Change: {change}"
    )
    assert get.call_args.args[0].endswith("instId=BTC-USDT")


@pytest.mark.parametrize("opening", ["0", "", None])
def test_unavailable_opening_price_keeps_current_price(
    monkeypatch: pytest.MonkeyPatch,
    tools: dict,
    opening: str | None,
) -> None:
    ticker = {"last": "110"}
    if opening is not None:
        ticker["open24h"] = opening
    mock_ticker(monkeypatch, ticker)
    assert tools["get_okx_crypto_price"]("BTC-USDT") == (
        "Current price of BTC/USDT: $110.00\n24h Change: N/A"
    )


def test_spot_volume_uses_quote_currency(
    monkeypatch: pytest.MonkeyPatch, tools: dict
) -> None:
    mock_ticker(
        monkeypatch,
        {"instType": "SPOT", "vol24h": "2", "volCcy24h": "120000"},
    )
    assert tools["get_okx_crypto_volume"]("BTC-USDT") == (
        "24h Trading Volume for BTC/USDT: 120,000.00 USDT"
    )


@pytest.mark.parametrize(
    "name", ["get_okx_crypto_price", "get_okx_crypto_volume"]
)
def test_api_error_is_preserved(
    monkeypatch: pytest.MonkeyPatch, tools: dict, name: str
) -> None:
    response = Mock()
    response.json.return_value = {
        "code": "51001",
        "msg": "Instrument does not exist",
        "data": [],
    }
    monkeypatch.setattr(requests, "get", Mock(return_value=response))
    assert (
        tools[name]("MISSING") == "Error: Instrument does not exist"
    )
