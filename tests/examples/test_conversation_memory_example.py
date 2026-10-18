"""Run the memory example with real Conversation persistence."""

import io
import socket
import sys
import unittest
import uuid
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "examples/changelogs/v12_update"
    / "04_conversation_memory_compact.py"
)
CORE = ROOT / "swarms/structs/conversation.py"


def run_memory_example(workspace):
    """Load the real module with offline dependency helpers."""
    swarms = ModuleType("swarms")
    structs = ModuleType("swarms.structs")
    utils = ModuleType("swarms.utils")
    tokenizer = ModuleType("swarms.utils.litellm_tokenizer")
    tokenizer.count_tokens = Mock(return_value=0)
    identifiers = ModuleType("swarms.utils.generate_id")
    identifiers.generate_id = lambda *args: uuid.uuid4().hex
    logging = ModuleType("loguru")
    logging.logger = Mock()
    conversation = ModuleType("swarms.structs.conversation")
    namespace = {"__name__": "offline_memory_example"}
    output = io.StringIO()
    with (
        patch.dict(
            sys.modules,
            {
                "swarms": swarms,
                "swarms.structs": structs,
                "swarms.utils": utils,
                "swarms.utils.litellm_tokenizer": tokenizer,
                "swarms.utils.generate_id": identifiers,
                "swarms.structs.conversation": conversation,
                "loguru": logging,
            },
        ),
        patch("tempfile.mkdtemp", return_value=str(workspace)),
        redirect_stdout(output),
    ):
        exec(  # noqa: S102 - actual local Conversation module
            compile(
                CORE.read_text(encoding="utf-8"), str(CORE), "exec"
            ),
            conversation.__dict__,
        )
        exec(  # noqa: S102 - actual local example
            compile(
                SOURCE.read_text(encoding="utf-8"),
                str(SOURCE),
                "exec",
            ),
            namespace,
        )
    return namespace, output.getvalue()


class ConversationMemoryExampleTests(unittest.TestCase):
    def setUp(self):
        network = patch.object(
            socket,
            "socket",
            side_effect=AssertionError("Network access is forbidden"),
        )
        network.start()
        self.addCleanup(network.stop)
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.workspace = Path(directory.name)

    def test_complete_example_reloads_persisted_messages(self):
        namespace, output = run_memory_example(self.workspace)
        reloaded = namespace["conv2"].return_history_as_string()
        self.assertIn("The capital of France is Paris.", reloaded)
        self.assertIn("12 million", reloaded)
        self.assertIn("Reloaded conversation history", output)
        self.assertEqual(
            namespace["conv"].memory_md_path,
            namespace["conv2"].memory_md_path,
        )

    def test_compaction_archives_original_and_persists_summary(self):
        namespace, _ = run_memory_example(self.workspace)
        memory = self.workspace / "agents/DemoAgent/MEMORY.md"
        archives = list(memory.parent.glob("archive/history_*.md"))
        self.assertEqual(len(archives), 1)
        self.assertIn(
            "The capital of France is Paris.",
            archives[0].read_text(encoding="utf-8"),
        )
        text = memory.read_text(encoding="utf-8")
        self.assertIn(namespace["summary_text"], text)
        self.assertNotIn("What is the capital of France?", text)
        history = namespace["conv"].return_history_as_string()
        self.assertIn(namespace["summary_text"], history)
        self.assertNotIn("What is the capital of France?", history)

    def test_claimed_timestamp_history_is_enabled(self):
        namespace, output = run_memory_example(self.workspace)
        self.assertTrue(namespace["conv"].time_enabled)
        self.assertIn("History with ISO timestamps", output)
        self.assertIn(
            "timestamp", namespace["conv"].conversation_history[-1]
        )


if __name__ == "__main__":
    unittest.main()
