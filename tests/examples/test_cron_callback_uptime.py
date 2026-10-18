"""Validate the transformation example with real CronJob metadata."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CronCallbackUptimeTests(unittest.TestCase):
    def setUp(self):
        self.cron = load_module(
            "callback_cron_under_test",
            ROOT / "swarms/structs/cron_job.py",
        )
        self.cron.logger = Mock()
        swarms = types.ModuleType("swarms")
        swarms.Agent = Mock()
        swarms.CronJob = self.cron.CronJob
        with patch.dict(sys.modules, {"swarms": swarms}):
            example = load_module(
                "callback_example_under_test",
                ROOT
                / "examples/guides/deployment/cron_job_examples"
                / "callback_cron_example.py",
            )
        self.job = self.cron.CronJob(
            agent=lambda task: "original output",
            interval="15seconds",
            callback=example.transform_output_callback,
        )

    def run_at(self, timestamp):
        with patch.object(
            self.cron.time, "time", return_value=timestamp
        ):
            return self.job._run_job("offline task")

    def test_running_job_keeps_transformed_output(self):
        self.job.start_time = 100
        self.job.is_running = True
        result = self.run_at(125)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["uptime_seconds"], 25)
        self.assertEqual(result["original_output"], "original output")
        self.assertEqual(result["task_executed"], "offline task")
        self.assertEqual(result["execution_number"], 1)
        self.assertEqual(result["job_status"], "running")
        self.cron.logger.warning.assert_not_called()

    def test_unstarted_job_has_zero_uptime(self):
        result = self.run_at(125)
        self.assertEqual(result["uptime_seconds"], 0)
        self.assertEqual(result["job_status"], "stopped")

    def test_uptime_advances_between_executions(self):
        self.job.start_time = 100
        first = self.run_at(125)
        second = self.run_at(140)
        self.assertIsInstance(first, dict)
        self.assertIsInstance(second, dict)
        self.assertEqual(first["uptime_seconds"], 25)
        self.assertEqual(second["uptime_seconds"], 40)
        self.assertEqual(second["execution_number"], 2)


if __name__ == "__main__":
    unittest.main()
