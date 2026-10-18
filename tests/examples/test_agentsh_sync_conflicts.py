"""Offline tests for unresolved conflicts in the AgentSH example."""

import runpy
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "examples/guides/agentsh_implementation/agentsh.py"
)


class AgentSHSyncConflictTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        swarms = ModuleType("swarms")
        swarms.Agent = Mock()
        with patch.dict(sys.modules, {"swarms": swarms}):
            cls.example = runpy.run_path(str(EXAMPLE))

    def setUp(self) -> None:
        self.org = self.example["AgentSH"](num_workers=1)
        self.workspace = self.org.workspace
        self.owner = self.org.handles[0]
        self.workspace.write("seed", "shared.txt", "base")
        self.assertTrue(self.workspace.merge("seed", "seed").merged)
        self.workspace.write(self.owner, "shared.txt", "mine")
        self.workspace.write("peer", "shared.txt", "theirs")
        self.assertTrue(
            self.workspace.merge("peer", "peer edit").merged
        )

    def test_repeated_sync_reports_unresolved_markers(self) -> None:
        self.assertEqual(
            self.workspace.sync(self.owner), ["shared.txt"]
        )
        original = self.workspace.read(self.owner, "shared.txt")
        self.assertIn("<<<<<<< yours", original)
        for _ in range(2):
            self.assertEqual(
                self.workspace.sync(self.owner), ["shared.txt"]
            )
            self.assertEqual(
                self.workspace.read(self.owner, "shared.txt"),
                original,
            )
        result = self.workspace.merge(self.owner, "still unresolved")
        self.assertFalse(result.merged)
        self.assertEqual(result.conflicts, ("shared.txt",))

    def test_sync_main_tool_keeps_reporting_the_conflict(
        self,
    ) -> None:
        worker = self.example["Worker"](self.owner, self.org)
        sync_main = next(
            tool
            for tool in worker._infrastructure_tools()
            if tool.__name__ == "sync_main"
        )
        for _ in range(2):
            message = sync_main()
            self.assertIn(
                "resolve conflict markers in: shared.txt", message
            )
            self.assertNotIn("no conflicts", message)

    def test_resolved_content_can_sync_and_merge(self) -> None:
        self.workspace.sync(self.owner)
        self.workspace.write(self.owner, "shared.txt", "resolved")
        self.assertEqual(self.workspace.sync(self.owner), [])
        self.assertTrue(
            self.workspace.merge(self.owner, "resolve").merged
        )
        self.assertEqual(
            self.workspace.snapshot()["shared.txt"], "resolved"
        )

    def test_resolving_by_deletion_can_sync_and_merge(self) -> None:
        self.workspace.sync(self.owner)
        self.workspace.delete(self.owner, "shared.txt")
        self.assertEqual(self.workspace.sync(self.owner), [])
        self.assertTrue(
            self.workspace.merge(self.owner, "delete").merged
        )
        self.assertNotIn("shared.txt", self.workspace.snapshot())

    def test_existing_and_new_conflicts_are_reported_once(
        self,
    ) -> None:
        self.workspace.sync(self.owner)
        original = self.workspace.read(self.owner, "shared.txt")
        self.workspace.write(self.owner, "second.txt", "mine")
        self.workspace.write("peer", "second.txt", "theirs")
        self.workspace.write(
            "peer", "unrelated.txt", "new upstream file"
        )
        self.assertTrue(
            self.workspace.merge("peer", "more edits").merged
        )
        self.assertEqual(
            self.workspace.sync(self.owner),
            ["shared.txt", "second.txt"],
        )
        self.assertEqual(
            self.workspace.read(self.owner, "shared.txt"), original
        )
        self.assertEqual(
            self.workspace.read(self.owner, "unrelated.txt"),
            "new upstream file",
        )


if __name__ == "__main__":
    unittest.main()
