import json
import runpy
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock

import pytest
import requests


@pytest.fixture
def coin_price(monkeypatch: pytest.MonkeyPatch) -> Callable[..., str]:
    """Load the example without running a real model agent."""
    swarms = ModuleType("swarms")
    swarms.Agent = Mock()
    monkeypatch.setitem(sys.modules, "swarms", swarms)
    example = (
        Path(__file__).resolve().parents[2]
        / "examples/tools/agent_as_tools.py"
    )
    return runpy.run_path(str(example))["get_coin_price"]


@pytest.mark.parametrize(
    ("args", "currency"),
    [(("bitcoin",), "usd"), (("bitcoin", "eur"), "eur")],
)
def test_coin_price_currency(
    coin_price: Callable[..., str],
    monkeypatch: pytest.MonkeyPatch,
    args: tuple[str, ...],
    currency: str,
) -> None:
    """Forward omitted and explicit currencies to the request."""
    data = {"bitcoin": {currency: 45000}}
    response = Mock()
    response.json.return_value = data
    request = Mock(return_value=response)
    monkeypatch.setattr(requests, "get", request)

    assert json.loads(coin_price(*args)) == data
    assert request.call_count == 1
    params = request.call_args.kwargs["params"]
    assert params["ids"] == "bitcoin"
    assert params["vs_currencies"] == currency
    response.raise_for_status.assert_called_once_with()


def test_coin_price_default_currency_request_error(
    coin_price: Callable[..., str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep request failures as JSON when the currency is omitted."""
    monkeypatch.setattr(
        requests,
        "get",
        Mock(side_effect=requests.ConnectionError("offline")),
    )

    assert json.loads(coin_price("bitcoin")) == {
        "error": "Failed to fetch price for bitcoin: offline"
    }
