import asyncio
import io
import runpy
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, PropertyMock, call, patch

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples/guides/demos"
EXAMPLE = EXAMPLE_DIR / "crypto/ethchain_agent.py"


class TestEthereumBlockMonitor(unittest.TestCase):
    def monitor(
        self,
        heights: list[int],
        block_effect: Exception | None = None,
        qualifies: bool = True,
    ) -> tuple:
        logger = Mock()
        imports = {
            "swarms": SimpleNamespace(Agent=Mock()),
            "web3": SimpleNamespace(Web3=Mock()),
            "loguru": SimpleNamespace(logger=logger),
            "dotenv": SimpleNamespace(load_dotenv=Mock()),
            "requests": SimpleNamespace(),
        }
        with (
            patch.dict(sys.modules, imports),
            patch("socket.socket.connect", side_effect=AssertionError),
            patch(
                "socket.socket.connect_ex",
                side_effect=AssertionError("No network"),
            ),
            patch(
                "socket.create_connection",
                side_effect=AssertionError("No network"),
            ),
            redirect_stdout(io.StringIO()),
        ):
            namespace = runpy.run_path(str(EXAMPLE))
            analyzer = object.__new__(namespace["EthereumAnalyzer"])
            analyzer.min_value_eth = 100
            analyzer.eth_price = 2
            analyzer.last_processed_block = 100
            eth = Mock()
            type(eth).block_number = PropertyMock(side_effect=heights)
            block = SimpleNamespace(
                transactions=[SimpleNamespace(hash="synthetic-hash")]
            )
            eth.get_block.return_value = block
            if block_effect is not None:
                eth.get_block.side_effect = [block_effect, block]
            analyzer.w3 = SimpleNamespace(eth=eth)
            analyzer.analyze_transaction = AsyncMock(
                return_value=(
                    {
                        "transaction_hash": "synthetic-hash",
                        "value_eth": 100,
                        "value_usd": 200,
                    }
                    if qualifies
                    else None
                )
            )
            analyzer.prepare_analysis_prompt = Mock(return_value="Analyze")
            analyzer.agent = Mock()
            analyzer.agent.run.return_value = "Offline analysis"
            analyzer.save_to_csv = Mock()
            processed = []

            async def sleep(_delay: float) -> None:
                processed.append(analyzer.last_processed_block)
                if len(processed) == len(heights):
                    raise asyncio.CancelledError

            with (
                patch.object(namespace["asyncio"], "sleep", side_effect=sleep),
                self.assertRaises(asyncio.CancelledError),
            ):
                asyncio.run(analyzer.monitor_transactions())
        imports["swarms"].Agent.assert_not_called()
        imports["dotenv"].load_dotenv.assert_not_called()
        return analyzer, processed, logger

    def test_unchanged_block_is_analyzed_and_saved_once(self) -> None:
        analyzer, processed, _ = self.monitor([101, 101, 101])
        analyzer.w3.eth.get_block.assert_called_once_with(
            101,
            full_transactions=True,
        )
        self.assertEqual(analyzer.agent.run.call_count, 1)
        self.assertEqual(analyzer.save_to_csv.call_count, 1)
        self.assertEqual(processed, [101, 101, 101])

    def test_new_block_is_processed_after_unchanged_height(self) -> None:
        analyzer, processed, _ = self.monitor([101, 101, 102])
        self.assertEqual(
            analyzer.w3.eth.get_block.call_args_list,
            [
                call(101, full_transactions=True),
                call(102, full_transactions=True),
            ],
        )
        self.assertEqual(analyzer.agent.run.call_count, 2)
        self.assertEqual(processed, [101, 101, 102])

    def test_initial_processed_height_is_skipped(self) -> None:
        analyzer, processed, _ = self.monitor([100, 100])
        analyzer.w3.eth.get_block.assert_not_called()
        analyzer.agent.run.assert_not_called()
        self.assertEqual(processed, [100, 100])

    def test_failed_fetch_retries_before_advancing_height(self) -> None:
        analyzer, processed, logger = self.monitor(
            [101, 101, 101], block_effect=RuntimeError("Offline unavailable")
        )
        self.assertEqual(analyzer.w3.eth.get_block.call_count, 2)
        self.assertEqual(analyzer.agent.run.call_count, 1)
        self.assertEqual(analyzer.save_to_csv.call_count, 1)
        self.assertEqual(processed, [100, 101, 101])
        logger.error.assert_called_once()

    def test_filtered_transactions_mark_the_block_processed(self) -> None:
        analyzer, processed, _ = self.monitor([101, 101], qualifies=False)
        self.assertEqual(analyzer.w3.eth.get_block.call_count, 1)
        analyzer.agent.run.assert_not_called()
        analyzer.save_to_csv.assert_not_called()
        self.assertEqual(processed, [101, 101])


if __name__ == "__main__":
    unittest.main()
