"""Test debate examples with real orchestration and formatting."""

import io
import socket
import sys
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples/multi_agent/debate_examples"


class OfflineAgent:
    def __init__(self, agent_name, **kwargs):
        self.agent_name = agent_name
        self.calls = []

    def run(self, task, **kwargs):
        self.calls.append(task)
        return f"{self.agent_name} output {len(self.calls)}"


def identity_trace(*args, **kwargs):
    return lambda function: function


def load_real_debate():
    names = [
        "swarms",
        "swarms.structs",
        "swarms.utils",
        "swarms.prompts",
        "swarms.structs.agent",
        "swarms.structs.conversation",
        "swarms.structs.context_utils",
        "swarms.structs.execution_utils",
        "swarms.structs.debate_with_judge",
        "swarms.utils.history_output_formatter",
        "swarms.utils.output_types",
        "swarms.prompts.debate_with_judge_prompts",
        "swarms.telemetry",
        "swarms.telemetry.otel",
        "swarms.utils.generate_id",
        "swarms.utils.litellm_tokenizer",
        "loguru",
    ]
    modules = {name: ModuleType(name) for name in names}
    modules["swarms.structs.agent"].Agent = OfflineAgent
    modules["swarms.utils.output_types"].HistoryOutputType = str
    telemetry = modules["swarms.telemetry.otel"]
    telemetry.ContextThreadPoolExecutor = ThreadPoolExecutor
    telemetry.capture_init = Mock()
    telemetry.trace_run = identity_trace
    modules["swarms.utils.generate_id"].generate_id = (
        lambda *args: uuid.uuid4().hex
    )
    modules["swarms.utils.litellm_tokenizer"].count_tokens = Mock(
        return_value=0
    )
    modules["loguru"].logger = Mock()
    with patch.dict(sys.modules, modules):
        for module_name in [
            "swarms.structs.conversation",
            "swarms.structs.context_utils",
            "swarms.structs.execution_utils",
            "swarms.utils.history_output_formatter",
            "swarms.prompts.debate_with_judge_prompts",
            "swarms.structs.debate_with_judge",
        ]:
            path = ROOT / (module_name.replace(".", "/") + ".py")
            exec(  # noqa: S102 - actual local source modules
                compile(
                    path.read_text(encoding="utf-8"),
                    str(path),
                    "exec",
                ),
                modules[module_name].__dict__,
            )
    return modules["swarms.structs.debate_with_judge"].DebateWithJudge


def run_example(filename, debate):
    swarms = ModuleType("swarms")
    swarms.Agent = OfflineAgent
    swarms.DebateWithJudge = debate
    namespace = {"__name__": "__main__"}
    path = EXAMPLES / filename
    output = io.StringIO()
    with patch.dict(sys.modules, {"swarms": swarms}), redirect_stdout(
        output
    ):
        exec(  # noqa: S102 - actual local example
            compile(
                path.read_text(encoding="utf-8"), str(path), "exec"
            ),
            namespace,
        )
    return namespace, output.getvalue()


class DebateExampleResultTests(unittest.TestCase):
    def setUp(self):
        network = patch.object(
            socket,
            "socket",
            side_effect=AssertionError("Network access is forbidden"),
        )
        network.start()
        self.addCleanup(network.stop)
        self.debate = load_real_debate()

    def test_business_example_reads_full_history_and_final_answer(
        self,
    ):
        namespace, output = run_example(
            "business_strategy_debate_example.py", self.debate
        )
        conversation = namespace["strategy_debate"].conversation
        self.assertEqual(len(conversation.conversation_history), 12)
        self.assertIn(
            "Growth-Strategy-Pro output 2", namespace["history"]
        )
        self.assertIn("Strategy-Judge output 5", namespace["history"])
        self.assertEqual(
            namespace["final_answer"], "Strategy-Judge output 5"
        )
        self.assertEqual(namespace["result"], conversation.to_dict())
        self.assertTrue(
            output.rstrip().endswith(namespace["final_answer"])
        )

    def test_policy_example_prints_the_last_judge_message(self):
        namespace, output = run_example(
            "policy_debate_example.py", self.debate
        )
        conversation = namespace["debate_system"].conversation
        self.assertEqual(len(conversation.conversation_history), 9)
        self.assertEqual(
            namespace["final_answer"], "Policy-Judge-Agent output 4"
        )
        self.assertIn(
            "Anti-Regulation-Agent output 2", namespace["result"]
        )
        self.assertIn(namespace["final_answer"], namespace["result"])
        self.assertTrue(
            output.rstrip().endswith(namespace["final_answer"])
        )

    def test_final_output_mode_returns_the_last_judge_message(
        self,
    ):
        agents = [
            OfflineAgent(name) for name in ["Pro", "Con", "Judge"]
        ]
        debate = self.debate(
            agents=agents, max_loops=1, output_type="final"
        )
        self.assertEqual(
            debate.run("Offline topic"), "Judge output 2"
        )


if __name__ == "__main__":
    unittest.main()
