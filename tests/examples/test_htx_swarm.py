"""Offline regression tests for the HTX example's recent-trade data."""

import runpy
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "examples/guides/demos/crypto/htx_swarm.py"
)


def run_example(batches: list) -> dict:
    """Execute the real example with HTTP, dotenv, and agents replaced."""
    requests = ModuleType("requests")
    requests.exceptions = SimpleNamespace(
        RequestException=ConnectionError
    )
    requests.get = Mock(
        side_effect=[
            Mock(json=Mock(return_value=payload))
            for payload in [
                {"status": "ok", "tick": {}},
                {"status": "ok", "tick": {}},
                {"status": "ok", "data": batches},
                {"status": "ok", "data": []},
            ]
        ]
    )
    dotenv = ModuleType("dotenv")
    dotenv.load_dotenv = Mock()
    swarms = ModuleType("swarms")
    swarms.Agent = Mock()
    swarms.SequentialWorkflow = Mock()
    with patch.dict(
        sys.modules,
        {"requests": requests, "dotenv": dotenv, "swarms": swarms},
    ), patch("builtins.print"):
        namespace = runpy.run_path(str(EXAMPLE))
    assert requests.get.call_count == 4
    return namespace["data"]


class HtxTradeTests(unittest.TestCase):
    def test_preserves_every_trade_in_timestamp_batches(self) -> None:
        """The documented one-then-two trade batches must yield three rows."""
        result = run_example(
            [
                {
                    "data": [
                        {
                            "id": 11,
                            "ts": 300,
                            "price": 100,
                            "amount": 1,
                            "direction": "buy",
                        }
                    ]
                },
                {
                    "data": [
                        {
                            "id": 12,
                            "ts": 200,
                            "price": 99,
                            "amount": 2,
                            "direction": "buy",
                        },
                        {
                            "id": 13,
                            "ts": 200,
                            "price": 98,
                            "amount": 3,
                            "direction": "sell",
                        },
                    ]
                },
            ]
        )
        self.assertEqual(
            result["recent_trades"],
            [
                {
                    "price": 100,
                    "amount": 1,
                    "direction": "buy",
                    "trade_id": 11,
                    "timestamp": 300,
                },
                {
                    "price": 99,
                    "amount": 2,
                    "direction": "buy",
                    "trade_id": 12,
                    "timestamp": 200,
                },
                {
                    "price": 98,
                    "amount": 3,
                    "direction": "sell",
                    "trade_id": 13,
                    "timestamp": 200,
                },
            ],
        )

    def test_single_trade_keeps_existing_field_mapping(self) -> None:
        """Single-trade batches retain the existing output shape."""
        result = run_example(
            [
                {
                    "data": [
                        {
                            "id": 1,
                            "ts": 2,
                            "price": 3,
                            "amount": 4,
                            "direction": "buy",
                        }
                    ]
                }
            ]
        )
        self.assertEqual(
            result["recent_trades"],
            [
                {
                    "price": 3,
                    "amount": 4,
                    "direction": "buy",
                    "trade_id": 1,
                    "timestamp": 2,
                }
            ],
        )

    def test_empty_trade_data(self) -> None:
        """Empty responses and empty timestamp batches contain no trades."""
        for batches in ([], [{"data": []}]):
            with self.subTest(batches=batches):
                self.assertEqual(
                    run_example(batches)["recent_trades"], []
                )


if __name__ == "__main__":
    unittest.main()
