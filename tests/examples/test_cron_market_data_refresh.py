"""Drive the market examples through real offline CronJob ticks."""

import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples/guides/deployment/cron_job_examples"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CronMarketDataRefreshTests(unittest.TestCase):
    def setUp(self):
        self.cron_module = load_module(
            "market_cron_under_test",
            ROOT / "swarms/structs/cron_job.py",
        )
        self.cron_module.logger = Mock()
        self.runner = Mock()
        self.runner.run.return_value = "offline analysis"
        swarms = types.ModuleType("swarms")
        swarms.Agent = Mock(return_value=self.runner)
        swarms.ConcurrentWorkflow = Mock(return_value=self.runner)
        swarms.CronJob = self.cron_module.CronJob
        modules = patch.dict(sys.modules, {"swarms": swarms})
        modules.start()
        self.addCleanup(modules.stop)
        request = patch("requests.get")
        self.get = request.start()
        self.addCleanup(request.stop)

    @staticmethod
    def response(price):
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(
            {
                "solana": {
                    "usd": price,
                    "usd_market_cap": 1000,
                    "usd_24h_vol": 100,
                    "usd_24h_change": 1,
                    "last_updated_at": 1,
                }
            }
        ).encode()
        return response

    def load_example(self, filename):
        module = load_module(
            "market_example_under_test", EXAMPLES / filename
        )
        module.logger = Mock()
        return module

    def run_two_ticks(self, module):
        jobs = []

        def start(job):
            jobs.append(job)
            job.schedule.run_all(delay_seconds=0)
            job.schedule.run_all(delay_seconds=0)

        with patch.object(self.cron_module.CronJob, "start", start):
            module.main()
        self.assertEqual(jobs[0].execution_count, 2)
        self.assertEqual(jobs[0].error_count, 0)
        return [
            call.kwargs["task"]
            for call in self.runner.run.call_args_list
        ]

    def assert_refreshed(self, filename):
        self.get.side_effect = [
            self.response(101),
            self.response(203),
        ]
        module = self.load_example(filename)
        tasks = self.run_two_ticks(module)
        self.assertEqual(self.get.call_count, 2)
        key = (
            "usd"
            if "simple_concurrent" in filename
            else "current_price_usd"
        )
        self.assertIn(f'"{key}": 101', tasks[0])
        self.assertIn(f'"{key}": 203', tasks[1])
        self.assertNotIn('": 101', tasks[1])

    def test_solana_fetches_current_data_on_every_tick(self):
        self.assert_refreshed("solana_price_tracker.py")

    def test_concurrent_crypto_fetches_current_data_on_every_tick(
        self,
    ):
        self.assert_refreshed("simple_concurrent_crypto_cron.py")
        for call in self.get.call_args_list:
            self.assertEqual(
                call.kwargs["params"]["ids"],
                "bitcoin,ethereum,solana",
            )

    def test_solana_import_does_not_fetch_a_stale_snapshot(self):
        self.get.return_value = self.response(101)
        self.load_example("solana_price_tracker.py")
        self.get.assert_not_called()

    def test_solana_retries_data_fetch_after_request_failure(self):
        self.get.side_effect = [
            requests.Timeout("offline timeout"),
            self.response(203),
        ]
        module = self.load_example("solana_price_tracker.py")
        tasks = self.run_two_ticks(module)
        self.assertIn("offline timeout", tasks[0])
        self.assertIn('"current_price_usd": 203', tasks[1])
        self.assertNotIn("offline timeout", tasks[1])


if __name__ == "__main__":
    unittest.main()
