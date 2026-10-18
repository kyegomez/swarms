"""Run the example with source-derived API mocks and no SDK imports."""

import ast
import io
import runpy
import sys
import typing
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, create_autospec, patch

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples/utils/litellm_network_error_handling.py"


class TestNetworkErrorExample(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        source = ROOT / "swarms/utils/litellm_wrapper.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        definitions = [
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name in ("LiteLLM", "NetworkConnectionError")
        ]
        namespace = dict(vars(typing))
        namespace.update(
            BaseModel=object,
            requests=SimpleNamespace(
                exceptions=SimpleNamespace(
                    ConnectionError=ConnectionError,
                    Timeout=TimeoutError,
                    RequestException=OSError,
                )
            ),
        )
        module = ast.Module(body=definitions, type_ignores=[])
        # Load API definitions without importing provider modules.
        code = compile(module, str(source), "exec")
        exec(code, namespace)  # noqa: S102
        cls.api = namespace["LiteLLM"]
        cls.network_error = namespace["NetworkConnectionError"]

    def run_example(
        self, responses: list | None = None, connected: bool = True
    ) -> tuple[Mock, str]:
        model = create_autospec(self.api, spec_set=True)
        model.check_internet_connection.return_value = connected
        model.return_value.run.return_value = "scripted response"
        model.return_value.run.side_effect = responses
        utils = ModuleType("swarms.utils")
        utils.LiteLLM = model
        utils.NetworkConnectionError = self.network_error
        modules = {
            "swarms": ModuleType("swarms"),
            "swarms.utils": utils,
        }
        output = io.StringIO()
        with (
            patch.dict(sys.modules, modules),
            patch(
                "socket.create_connection",
                side_effect=AssertionError(
                    "Network access is forbidden"
                ),
            ),
            redirect_stdout(output),
        ):
            runpy.run_path(str(EXAMPLE), run_name="__main__")
        return model, output.getvalue()

    def test_main_completes_all_supported_examples(self) -> None:
        model, output = self.run_example()
        self.assertEqual(model.call_count, 3)
        self.assertEqual(model.return_value.run.call_count, 3)
        self.assertIn("Example 3: Custom Base URL", output)
        self.assertEqual(
            model.call_args.kwargs["base_url"],
            "http://localhost:11434",
        )

    def test_network_failure_keeps_local_fallback(self) -> None:
        model, output = self.run_example(
            [
                self.network_error("offline"),
                "local answer",
                "ok",
                "ok",
            ],
            connected=False,
        )
        self.assertIn("Local model response: local answer", output)
        self.assertEqual(model.return_value.run.call_count, 4)
        names = [
            call.kwargs["model_name"] for call in model.call_args_list
        ]
        self.assertEqual(names, ["gpt-5.4"] + ["ollama/llama2"] * 3)

    def test_unavailable_servers_keep_troubleshooting_output(
        self,
    ) -> None:
        _, output = self.run_example(
            [self.network_error("offline") for _ in range(4)],
            connected=False,
        )
        self.assertIn(
            "Make sure Ollama is installed and running", output
        )
        self.assertIn("Error: offline", output)
        self.assertIn("Troubleshooting:", output)


if __name__ == "__main__":
    unittest.main()
