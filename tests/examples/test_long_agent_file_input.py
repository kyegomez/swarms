import ast
import concurrent.futures
import io
import os
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Union
from unittest.mock import Mock, patch

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples"
EXAMPLE = (
    EXAMPLE_DIR / "multi_agent/paper_implementations/long_agent.py"
)


class OfflineAgent:
    def __init__(self, **kwargs) -> None:
        self.agent_name = kwargs["agent_name"]
        self.run = Mock(return_value="Offline document summary")


class TestLongAgentFileInput(unittest.TestCase):
    def setUp(self) -> None:
        source = ast.parse(EXAMPLE.read_text())
        definition = next(
            node
            for node in source.body
            if isinstance(node, ast.ClassDef)
            and node.name == "LongAgent"
        )
        self.count_tokens = Mock(
            side_effect=lambda text: len(text.split())
        )
        namespace = {
            "concurrent": concurrent,
            "os": os,
            "Path": Path,
            "Union": Union,
            "List": list,
            "Agent": OfflineAgent,
            "Conversation": Mock(),
            "formatter": Mock(),
            "count_tokens": self.count_tokens,
            "markdown": Mock(),
        }
        exec(  # noqa: S102 - Load the class with offline dependencies.
            compile(
                ast.Module(body=[definition], type_ignores=[]),
                str(EXAMPLE),
                "exec",
            ),
            namespace,
        )
        self.agent = namespace["LongAgent"]()
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "document.txt"
        self.content = (
            "Actual document text with unique Unicode 内容."
        )
        self.path.write_text(self.content, encoding="utf-8")

    def count_document(self, path: str | Path) -> int:
        with (
            redirect_stdout(io.StringIO()),
            patch("socket.socket", side_effect=AssertionError),
        ):
            return self.agent.count_token_document(path)

    def test_string_path_counts_file_contents(self) -> None:
        self.assertEqual(
            self.count_document(str(self.path)),
            len(self.content.split()),
        )
        self.count_tokens.assert_called_once_with(self.content)

    def test_path_object_counts_file_contents(self) -> None:
        self.assertEqual(
            self.count_document(self.path), len(self.content.split())
        )

    def test_text_file_contents_reach_document_agent(self) -> None:
        with patch("socket.socket", side_effect=AssertionError):
            agents = self.agent.create_agents_for_documents(
                [self.path]
            )
        self.assertEqual(len(agents), 1)
        prompt = agents[0].run.call_args.args[0]
        self.assertIn(self.content, prompt)
        self.assertNotIn(str(self.path), prompt)

    def test_missing_file_raises_before_tokenization(self) -> None:
        with self.assertRaises(FileNotFoundError):
            self.count_document(
                str(self.path.with_name("missing.txt"))
            )
        self.count_tokens.assert_not_called()

    def test_markdown_dispatch_still_reads_contents(self) -> None:
        markdown_path = self.path.with_suffix(".md")
        markdown_path.write_text(self.content, encoding="utf-8")
        self.assertEqual(
            self.count_document(str(markdown_path)),
            len(self.content.split()),
        )

    def test_load_text_keeps_literal_text_contract(self) -> None:
        self.assertEqual(
            self.agent.load_text(self.content), self.content
        )
        self.assertEqual(self.agent.get_content(), self.content)


if __name__ == "__main__":
    unittest.main()
