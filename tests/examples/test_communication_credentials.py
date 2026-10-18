"""Import communication examples against a synthetic environment."""

import socket
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "examples/multi_agent/utils/agent_communication_examples.py"
)


def import_with_environment(values):
    """Never read or replace the process's actual environment."""
    fake_os = ModuleType("os")
    fake_os.environ = values
    fake_swarms = ModuleType("swarms")
    fake_swarms.Agent = Mock(name="OfflineAgent")
    namespace = {"__name__": "offline_communication_example"}
    with patch.dict(
        sys.modules, {"os": fake_os, "swarms": fake_swarms}
    ):
        exec(  # noqa: S102 - execute only the local example
            compile(
                SOURCE.read_text(encoding="utf-8"),
                str(SOURCE),
                "exec",
            ),
            namespace,
        )
    return namespace, fake_swarms.Agent


class CommunicationCredentialTests(unittest.TestCase):
    def setUp(self):
        network = patch.object(
            socket,
            "socket",
            side_effect=AssertionError("Network access is forbidden"),
        )
        network.start()
        self.addCleanup(network.stop)

    def test_import_preserves_an_existing_configuration(self):
        values = {"OPENAI_API_KEY": "synthetic-test-value"}
        original = values.copy()
        import_with_environment(values)
        self.assertEqual(values, original)

    def test_import_does_not_create_an_invalid_placeholder(self):
        values = {}
        import_with_environment(values)
        self.assertEqual(values, {})

    def test_import_preserves_other_values_and_does_not_run_agents(
        self,
    ):
        values = {"OFFLINE_FIXTURE": "unchanged"}
        namespace, agent = import_with_environment(values)
        self.assertEqual(values["OFFLINE_FIXTURE"], "unchanged")
        self.assertTrue(
            callable(
                namespace["example_1_direct_agent_communication"]
            )
        )
        self.assertTrue(callable(namespace["main"]))
        agent.assert_not_called()


if __name__ == "__main__":
    unittest.main()
