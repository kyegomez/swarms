"""Run the Figma comparison example with offline provider doubles."""

import contextlib
import importlib.util
import io
import json
import runpy
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

EXAMPLES = (
    Path(__file__).resolve().parents[2]
    / "examples/guides/deployment/cron_job_examples"
)
HELPER = EXAMPLES / "cron_job_example.py"
COMPARISON = EXAMPLES / "cron_job_figma_stock_swarms_tools_example.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FigmaComparisonHelperTests(unittest.TestCase):
    def setUp(self):
        ticker = Mock()
        ticker.info = {"longName": "Figma", "currentPrice": 50}
        ticker.history.return_value = types.SimpleNamespace(
            empty=True
        )
        ticker.fast_info = types.SimpleNamespace()
        self.yfinance = types.ModuleType("yfinance")
        self.yfinance.Ticker = Mock(return_value=ticker)
        swarms = types.ModuleType("swarms")
        swarms.Agent = Mock()
        swarms.Agent.return_value.run.return_value = (
            "offline analysis"
        )
        finance = types.ModuleType(
            "swarms.prompts.finance_agent_sys_prompt"
        )
        finance.FINANCIAL_AGENT_SYS_PROMPT = "offline prompt"
        swarms_tools = types.ModuleType("swarms_tools")
        swarms_tools.yahoo_finance_api = Mock(
            return_value={"symbol": "FIG"}
        )
        modules = patch.dict(
            sys.modules,
            {
                "yfinance": self.yfinance,
                "swarms": swarms,
                "swarms.prompts.finance_agent_sys_prompt": finance,
                "swarms_tools": swarms_tools,
            },
        )
        modules.start()
        self.addCleanup(modules.stop)
        self.output = io.StringIO()
        capture = contextlib.redirect_stdout(self.output)
        capture.__enter__()
        self.addCleanup(capture.__exit__, None, None, None)

    def test_comparison_runs_the_real_custom_helper_once(self):
        example = load_module(
            "figma_comparison_under_test", COMPARISON
        )
        example.logger = Mock()
        with patch.dict(sys.modules):
            sys.modules.pop("cron_job_example", None)
            with patch.object(
                sys, "path", [str(EXAMPLES), *sys.path]
            ):
                example.main()
        self.assertIn(
            "Custom function output:", self.output.getvalue()
        )
        self.yfinance.Ticker.assert_called_once_with("FIG")
        example.logger.info.assert_any_call(
            "All methods completed successfully!"
        )
        example.logger.error.assert_not_called()

    def test_importing_helper_does_not_request_market_data(self):
        load_module("figma_helper_under_test", HELPER)
        self.yfinance.Ticker.assert_not_called()
        self.assertEqual(self.output.getvalue(), "")

    def test_custom_helper_still_returns_stock_json(self):
        helper = load_module("figma_helper_under_test", HELPER)
        self.yfinance.Ticker.reset_mock()
        data = json.loads(helper.get_figma_stock_data("FIG"))
        self.yfinance.Ticker.assert_called_once_with("FIG")
        self.assertEqual(data["company_info"]["symbol"], "FIG")
        self.assertEqual(
            data["current_market_data"]["current_price"], 50
        )

    def test_running_helper_as_script_still_prints_stock_json(self):
        runpy.run_path(str(HELPER), run_name="__main__")
        data = json.loads(self.output.getvalue())
        self.yfinance.Ticker.assert_called_once_with("FIG")
        self.assertEqual(data["company_info"]["name"], "Figma")


if __name__ == "__main__":
    unittest.main()
