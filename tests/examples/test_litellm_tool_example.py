"""Exercise the tool example's real schema and wrapper offline."""

import runpy
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = (
    ROOT
    / "examples/single_agent/capabilities/tools"
    / "litellm_tool_example.py"
)


class ToolFunction(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: ToolFunction


class LiteLLMToolExampleTests(unittest.TestCase):
    def setUp(self):
        modules = patch.dict(sys.modules)
        modules.start()
        self.addCleanup(modules.stop)
        for name in list(sys.modules):
            if name == "swarms" or name.startswith("swarms."):
                del sys.modules[name]
        for name in ("swarms", "swarms.tools", "swarms.utils"):
            module = types.ModuleType(name)
            module.__path__ = [str(ROOT.joinpath(*name.split(".")))]
            sys.modules[name] = module

        self.tool_calls = [
            ToolCall(
                id="call_offline",
                function=ToolFunction(
                    name="get_stock_data",
                    arguments='{"symbol":"AAPL"}',
                ),
            )
        ]
        response = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content=None, tool_calls=self.tool_calls
                    ),
                    finish_reason="tool_calls",
                )
            ]
        )
        self.backend = types.ModuleType("swarms.utils.llm_backend")
        self.backend.BACKEND = "offline"
        self.backend.completion = Mock(return_value=response)
        self.backend.acompletion = Mock()
        self.backend.supports_reasoning = Mock(return_value=False)
        self.backend.supports_vision = Mock(return_value=False)
        sys.modules[self.backend.__name__] = self.backend
        logging = types.ModuleType("swarms.utils.loguru_logger")
        logging.initialize_logger = Mock(return_value=Mock())
        sys.modules[logging.__name__] = logging
        formatter = types.ModuleType("swarms.utils.formatter")
        formatter.formatter = Mock()
        sys.modules[formatter.__name__] = formatter
        for target in (
            "requests.sessions.Session.request",
            "socket.create_connection",
        ):
            network = patch(
                target,
                side_effect=AssertionError("Network is disabled"),
            )
            network.start()
            self.addCleanup(network.stop)

    def execute(self):
        with patch("builtins.print") as printed:
            namespace = runpy.run_path(str(EXAMPLE))
        return namespace, printed

    def test_request_contains_a_complete_function_tool(self):
        self.execute()
        params = self.backend.completion.call_args.kwargs
        tool = params["tools"][0]
        self.assertEqual(tool["type"], "function")
        function = tool["function"]
        self.assertEqual(function["name"], "get_stock_data")
        self.assertEqual(
            function["parameters"]["required"], ["symbol"]
        )
        self.assertEqual(
            function["parameters"]["properties"]["symbol"]["type"],
            "string",
        )
        self.assertIn("AAPL", params["messages"][-1]["content"])

    def test_example_prints_returned_tool_calls_instead_of_none(self):
        _, printed = self.execute()
        printed.assert_called_once_with(self.tool_calls)

    def test_followup_run_retains_tools_and_tool_call_output(self):
        namespace, _ = self.execute()
        first_tools = self.backend.completion.call_args.kwargs[
            "tools"
        ]
        output = namespace["llm"].run("Get MSFT data next")
        self.assertEqual(output, self.tool_calls)
        self.assertEqual(
            self.backend.completion.call_args.kwargs["tools"],
            first_tools,
        )


if __name__ == "__main__":
    unittest.main()
