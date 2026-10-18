"""Read bundled judging input from any launch directory."""

import contextlib
import io
import os
import runpy
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

EXAMPLES = (
    Path(__file__).resolve().parents[2]
    / "examples/guides/hackathon_judge_agent"
)
SCRIPT = EXAMPLES / "hackathon_judger_agent.py"


@contextlib.contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class HackathonProjectsPathTests(unittest.TestCase):
    def setUp(self):
        self.agent = Mock()
        self.agent.run.return_value = "offline ranking"
        swarms = types.ModuleType("swarms")
        swarms.Agent = Mock(return_value=self.agent)
        modules = patch.dict(sys.modules, {"swarms": swarms})
        modules.start()
        self.addCleanup(modules.stop)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cwd = Path(self.directory.name)
        self.bundled = (EXAMPLES / "projects.csv").read_text(
            encoding="utf-8"
        )

    def execute(self, directory):
        with (
            working_directory(directory),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            return runpy.run_path(str(SCRIPT), run_name="__main__")

    def test_runs_from_a_directory_without_projects_csv(self):
        self.execute(self.cwd)
        self.agent.run.assert_called_once_with(task=self.bundled)

    def test_default_ignores_an_unrelated_projects_csv(self):
        (self.cwd / "projects.csv").write_text(
            "name,description\nunrelated,wrong input\n",
            encoding="utf-8",
        )
        self.execute(self.cwd)
        self.agent.run.assert_called_once_with(task=self.bundled)

    def test_explicit_relative_path_still_uses_callers_directory(
        self,
    ):
        namespace = self.execute(EXAMPLES)
        custom = "name,description\n测试,custom input\n"
        (self.cwd / "custom.csv").write_text(custom, encoding="utf-8")
        with working_directory(self.cwd):
            content = namespace["read_csv_file"]("custom.csv")
        self.assertEqual(content, custom)

    def test_missing_explicit_file_still_raises(self):
        namespace = self.execute(EXAMPLES)
        with (
            working_directory(self.cwd),
            self.assertRaises(FileNotFoundError),
        ):
            namespace["read_csv_file"]("missing.csv")


if __name__ == "__main__":
    unittest.main()
