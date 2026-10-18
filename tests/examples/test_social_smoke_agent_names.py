"""Run the smoke example with real SocialAlgorithms state changes."""

import io
import socket
import sys
import unittest
import uuid
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType
from typing import ClassVar
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "examples/multi_agent/social_algorithms_examples"
    / "refactor_smoke_example.py"
)


class OfflineAgent:
    instances: ClassVar[list] = []

    def __init__(self, agent_name, **kwargs):
        self.agent_name = agent_name
        self.calls = []
        self.instances.append(self)

    def run(self, task):
        self.calls.append(task)
        return f"{self.agent_name}-output"

    def talk_to(self, other, task):
        return other.run(task)


def identity_trace(*args, **kwargs):
    return lambda function: function


def run_example(main):
    names = [
        "swarms",
        "swarms.structs",
        "swarms.utils",
        "swarms.structs.agent",
        "swarms.structs.conversation",
        "swarms.structs.social_algorithms",
        "swarms.structs.omni_agent_types",
        "swarms.utils.loguru_logger",
        "swarms.utils.output_types",
        "swarms.telemetry",
        "swarms.telemetry.otel",
        "swarms.utils.generate_id",
        "swarms.utils.litellm_tokenizer",
        "loguru",
    ]
    modules = {name: ModuleType(name) for name in names}
    modules["swarms"].Agent = OfflineAgent
    modules["swarms.structs.agent"].Agent = OfflineAgent
    modules["swarms.structs.omni_agent_types"].AgentType = (
        OfflineAgent
    )
    modules["swarms.utils.output_types"].OutputType = str
    modules["swarms.utils.loguru_logger"].initialize_logger = (
        lambda **kwargs: Mock()
    )
    telemetry = modules["swarms.telemetry.otel"]
    telemetry.capture_init = Mock()
    telemetry.trace_run = identity_trace
    modules["swarms.utils.generate_id"].generate_id = (
        lambda *args: uuid.uuid4().hex
    )
    modules["swarms.utils.litellm_tokenizer"].count_tokens = Mock(
        return_value=0
    )
    modules["loguru"].logger = Mock()
    output = io.StringIO()
    namespace = {"__name__": "__main__" if main else "offline_smoke"}
    with patch.dict(sys.modules, modules), redirect_stdout(output):
        for filename, module_name in [
            ("conversation.py", "swarms.structs.conversation"),
            (
                "social_algorithms.py",
                "swarms.structs.social_algorithms",
            ),
        ]:
            path = ROOT / "swarms/structs" / filename
            exec(  # noqa: S102 - actual local modules
                compile(
                    path.read_text(encoding="utf-8"),
                    str(path),
                    "exec",
                ),
                modules[module_name].__dict__,
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


class SocialSmokeAgentNamesTests(unittest.TestCase):
    def setUp(self):
        OfflineAgent.instances = []
        network = patch.object(
            socket,
            "socket",
            side_effect=AssertionError("Network access is forbidden"),
        )
        network.start()
        self.addCleanup(network.stop)

    def test_main_reports_names_after_membership_changes(self):
        _, output = run_example(True)
        self.assertIn(
            "agent names: ['Researcher', 'Analyst', 'Writer']", output
        )
        after_add = ["Researcher", "Analyst", "Writer", "Reviewer"]
        self.assertIn(f"after add: {after_add}", output)
        self.assertIn(
            "after remove: ['Researcher', 'Analyst', 'Writer']",
            output,
        )

    def test_main_runs_the_pipeline_after_removing_the_reviewer(self):
        _, output = run_example(True)
        agents = {
            agent.agent_name: agent
            for agent in OfflineAgent.instances
        }
        self.assertEqual(agents["Reviewer"].calls, [])
        self.assertIn("Researcher-output", agents["Analyst"].calls[0])
        self.assertIn("Analyst-output", agents["Writer"].calls[0])
        self.assertIn("Writer-output", output)

    def test_custom_algorithm_still_returns_each_stage_output(self):
        namespace, _ = run_example(False)
        agents = [OfflineAgent(name) for name in ["A", "B", "C"]]
        result = namespace["research_analysis_writeup"](
            agents, "Task"
        )
        self.assertEqual(
            result,
            {
                "research": "A-output",
                "analysis": "B-output",
                "writeup": "C-output",
            },
        )


if __name__ == "__main__":
    unittest.main()
