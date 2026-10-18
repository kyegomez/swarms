"""Exercise the example against the documented DEX Screener schema."""

import asyncio
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "examples/guides/demos/crypto/swarms_coin_multimarket.py"
)
TOKEN = "74SBV4zDXxTRgv1pEMoECskKBkZHc2yGPnc7GYVepump"
URL = f"https://api.dexscreener.com/token-pairs/v1/solana/{TOKEN}"


@pytest.fixture
def fetcher(monkeypatch: pytest.MonkeyPatch):
    blocked = Mock(side_effect=AssertionError("No network calls"))
    monkeypatch.setitem(
        sys.modules, "aiohttp", SimpleNamespace(ClientSession=blocked)
    )
    monkeypatch.setitem(
        sys.modules,
        "swarms.structs.agent",
        SimpleNamespace(Agent=Mock()),
    )
    return runpy.run_path(str(EXAMPLE))["MultiExchangeDataFetcher"]()


def test_uses_token_address_and_preserves_market_data(fetcher):
    pair = {
        "baseToken": {"address": TOKEN},
        "priceUsd": "0.125",
        "volume": {"h24": 1234.5, "h1": 15.0},
        "marketCap": 50000,
    }
    fetcher.fetch_data = AsyncMock(return_value=[pair])
    result = asyncio.run(fetcher.get_dexscreener_data())

    fetcher.fetch_data.assert_awaited_once_with(URL)
    assert result == {
        "price": 0.125,
        "volume24h": 1234.5,
        "marketCap": 50000.0,
    }


def test_does_not_report_the_quote_tokens_price(fetcher):
    pairs = [
        {
            "baseToken": {"address": "another-token"},
            "quoteToken": {"address": TOKEN},
            "priceUsd": "100.0",
        },
        {
            "baseToken": {"address": TOKEN},
            "priceUsd": "0.25",
            "volume": {"h24": 0},
            "marketCap": 1000,
        },
    ]
    fetcher.fetch_data = AsyncMock(return_value=pairs)
    result = asyncio.run(fetcher.get_dexscreener_data())
    assert result == {
        "price": 0.25,
        "volume24h": 0.0,
        "marketCap": 1000.0,
    }


def test_nullable_fields_are_not_fabricated_as_zero(fetcher):
    fetcher.fetch_data = AsyncMock(
        return_value=[
            {
                "baseToken": {"address": TOKEN},
                "priceUsd": None,
                "volume": {},
                "marketCap": None,
            }
        ]
    )
    result = asyncio.run(fetcher.get_dexscreener_data())
    assert result == {
        "price": None,
        "volume24h": None,
        "marketCap": None,
    }
    aggregated = fetcher.aggregate_data([result])
    assert aggregated["price"]["sources"] == 0
    assert aggregated["volume_24h"]["sources"] == 0
    assert aggregated["market_cap"]["sources"] == 0


@pytest.mark.parametrize(
    "response",
    [None, [], [{"baseToken": {"address": "another-token"}}]],
)
def test_unavailable_token_data_returns_none(fetcher, response):
    fetcher.fetch_data = AsyncMock(return_value=response)
    assert asyncio.run(fetcher.get_dexscreener_data()) is None
