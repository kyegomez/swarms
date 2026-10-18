import io
import runpy
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from unittest.mock import Mock, patch

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples/guides/demos"
EXAMPLE = EXAMPLE_DIR / "cuda_swarm.py"


class TestCudaSwarmOutput(unittest.TestCase):
    def assert_saved_output(self, response: str, expected: str) -> None:
        swarms_stub = ModuleType("swarms")
        swarms_stub.Agent = Mock()
        agent = Mock(spec=["run"])
        agent.run.return_value = response
        with (
            patch.dict(
                sys.modules,
                {"swarms": swarms_stub, "litellm": ModuleType("litellm")},
            ),
            patch("socket.socket", side_effect=AssertionError("No network")),
            patch(
                "socket.create_connection",
                side_effect=AssertionError("No network"),
            ),
            TemporaryDirectory() as directory,
            redirect_stdout(io.StringIO()),
        ):
            namespace = runpy.run_path(str(EXAMPLE))
            path = Path(directory) / "kernel.cu"
            swarm = namespace["CudaSwarm"](agents=[agent], file_path=str(path))
            result = swarm.run("Translate the kernel")
            self.assertEqual(path.read_bytes(), expected.encode())
            self.assertEqual(result, expected)
        agent.run.assert_called_once_with("Translate the kernel")

    def test_plain_cuda_is_saved(self) -> None:
        code = "__global__ void kernel() {}\n"
        self.assert_saved_output(code, code)

    def test_cuda_fence_is_removed(self) -> None:
        self.assert_saved_output(
            "```cuda\n__global__ void kernel() {}\n```",
            "__global__ void kernel() {}\n",
        )

    def test_cplusplus_fence_is_removed(self) -> None:
        self.assert_saved_output(
            "```c++\n__global__ void kernel() {}\n```",
            "__global__ void kernel() {}\n",
        )

    def test_unlabeled_fence_is_removed(self) -> None:
        self.assert_saved_output("```\nint value;\n```", "int value;\n")

    def test_multiple_blocks_exclude_prose(self) -> None:
        self.assert_saved_output(
            "Kernel:\n```cuda\nvoid kernel();\n```\n"
            "Launcher:\n```cpp\nint main() {}\n```\nDone.",
            "void kernel();\n\nint main() {}\n",
        )

    def test_crlf_fence_is_removed(self) -> None:
        response = "```cuda\r\nint value;\r\n```"
        self.assert_saved_output(response, "int value;\r\n")

    def test_empty_response_is_saved(self) -> None:
        self.assert_saved_output("", "")

    def test_empty_fence_remains_empty(self) -> None:
        self.assert_saved_output("```cuda\n```", "")


if __name__ == "__main__":
    unittest.main()
