"""Exercise FastAPI agent routes without calling a model provider."""

import asyncio
import importlib.util
import sys
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "examples/guides/deployment/fastapi"
    / "fastapi_agent_api_example.py"
)
HAS_DEPENDENCIES = all(
    importlib.util.find_spec(name) is not None
    for name in ("fastapi", "httpx", "uvicorn", "loguru")
)


@unittest.skipUnless(
    HAS_DEPENDENCIES, "Install the FastAPI example dependencies"
)
class FastAPIAgentConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.agent = Mock()
        swarms = types.ModuleType("swarms")
        swarms.Agent = Mock(return_value=self.agent)
        spec = importlib.util.spec_from_file_location(
            "fastapi_agent_example_under_test", EXAMPLE
        )
        self.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"swarms": swarms}):
            spec.loader.exec_module(self.module)
        self.module.logger = Mock()

    def client(self):
        import httpx

        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.module.app),
            base_url="http://example.test",
            trust_env=False,
        )

    async def assert_health_during_task(self, route, result_key):
        started = threading.Event()
        release = threading.Event()
        finished = threading.Event()

        def run_agent(*, task):
            started.set()
            release.wait(timeout=2)
            finished.set()
            return "offline result"

        self.agent.run.side_effect = run_agent
        async with self.client() as client:
            pending = asyncio.create_task(
                client.post(route, json={"task": "offline task"})
            )
            try:
                self.assertTrue(
                    await asyncio.to_thread(started.wait, 3)
                )
                health = await asyncio.wait_for(
                    client.get("/health"), timeout=3
                )
                self.assertEqual(health.status_code, 200)
                self.assertEqual(health.json()["status"], "healthy")
                self.assertFalse(
                    finished.is_set(),
                    "Health was blocked until agent.run completed",
                )
            finally:
                release.set()
                response = await pending
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()[result_key], "offline result"
        )
        self.agent.run.assert_called_once_with(task="offline task")

    async def assert_task_error(self, route, prefix):
        self.agent.run.side_effect = ValueError("offline failure")
        async with self.client() as client:
            response = await client.post(
                route, json={"task": "offline task"}
            )
        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json()["detail"], f"{prefix}: offline failure"
        )

    async def test_run_does_not_block_health(self):
        await self.assert_health_during_task("/agent/run", "result")

    async def test_chat_does_not_block_health(self):
        await self.assert_health_during_task(
            "/agent/chat", "response"
        )

    async def test_quantitative_trading_does_not_block_health(self):
        await self.assert_health_during_task(
            "/agent/quantitative-trading", "result"
        )

    async def test_run_preserves_errors(self):
        await self.assert_task_error(
            "/agent/run", "Agent execution failed"
        )

    async def test_chat_preserves_errors(self):
        await self.assert_task_error("/agent/chat", "Chat failed")

    async def test_quantitative_trading_preserves_errors(self):
        await self.assert_task_error(
            "/agent/quantitative-trading",
            "Quantitative trading agent failed",
        )


if __name__ == "__main__":
    unittest.main()
