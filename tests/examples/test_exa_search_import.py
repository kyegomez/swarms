"""Run the Exa example with the real optional tool and offline HTTP."""

import importlib
import importlib.util
import io
import json
import os
import runpy
import sys
import types
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import httpx

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(os.environ.get("SWARMS_SOURCE_ROOT", ROOT))
EXAMPLE = ROOT / (
    "examples/tools/base_tool_examples/" "exa_search_deep_research.py"
)


@unittest.skipUnless(
    importlib.util.find_spec("swarms_tools"),
    "Install swarms-tools to run this optional integration test",
)
class ExaSearchExampleTests(unittest.TestCase):
    """Exercise the public tool export, request and printed result."""

    @classmethod
    def setUpClass(cls) -> None:
        """Keep native dependency imports outside module-cache patches."""
        for name in ("dotenv", "loguru", "lxml.html", "orjson"):
            importlib.import_module(name)

    def run_example(
        self,
        status: int = 200,
        api_key: str | None = "offline-placeholder",
    ) -> tuple[str, list[httpx.Request]]:
        """Execute the full script without dotenv reads or live HTTP."""
        requests = []
        response_data = {
            "results": [
                {"title": "Frameworks", "url": "https://example.com"}
            ]
        }

        def respond(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(status, json=response_data)

        packages = {}
        for name in ("swarms", "swarms.structs", "swarms.utils"):
            module = types.ModuleType(name)
            module.__path__ = [
                str(SOURCE_ROOT / name.replace(".", "/"))
            ]
            packages[name] = module

        output = io.StringIO()
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, packages))
            stack.enter_context(
                patch("dotenv.load_dotenv", return_value=False)
            )
            stack.enter_context(
                patch("os.getenv", return_value=api_key)
            )
            stack.enter_context(
                patch(
                    "socket.socket.connect",
                    side_effect=AssertionError("Network disabled"),
                )
            )
            client = stack.enter_context(
                httpx.Client(
                    transport=httpx.MockTransport(respond),
                    trust_env=False,
                )
            )
            stack.enter_context(
                patch("httpx.post", side_effect=client.post)
            )
            stack.enter_context(redirect_stdout(output))
            runpy.run_path(str(EXAMPLE), run_name="__main__")
        return output.getvalue(), requests

    def test_search_prints_real_tool_result(self) -> None:
        output, requests = self.run_example()
        self.assertEqual(len(requests), 1)
        self.assertEqual(
            str(requests[0].url), "https://api.exa.ai/search"
        )
        self.assertEqual(
            json.loads(requests[0].content)["query"],
            "What are the best multi-agent frameworks ",
        )
        self.assertIn('title: "Frameworks"', output)
        self.assertIn('url: "https://example.com"', output)

    def test_http_failure_is_reported_by_real_tool(self) -> None:
        output, requests = self.run_example(status=503)
        self.assertEqual(len(requests), 1)
        self.assertIn("Search failed:", output)
        self.assertIn("503", output)

    def test_missing_key_raises_tool_configuration_error(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "EXA_API_KEY"):
            self.run_example(api_key=None)


if __name__ == "__main__":
    unittest.main()
