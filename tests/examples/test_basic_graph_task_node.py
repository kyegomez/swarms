"""Run the fan-in example with actual graph and conversation code."""

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
SOURCE = (
    ROOT
    / "examples/multi_agent/graphworkflow_examples/graph"
    / "graph_workflow_basic.py"
)


class OfflineAgent:
    def __init__(self, agent_name, **kwargs):
        self.agent_name = agent_name
        self.calls = []

    def run(self, task, messages=None, **kwargs):
        self.calls.append((task, messages or []))
        return f"Offline output from {self.agent_name}"


def identity_trace(*args, **kwargs):
    return lambda function: function


def load_real_graph():
    """Load real modules with offline dependency helpers."""
    names = [
        "swarms",
        "swarms.structs",
        "swarms.utils",
        "swarms.structs.agent",
        "swarms.structs.conversation",
        "swarms.utils.get_cpu_cores",
        "swarms.utils.loguru_logger",
        "swarms.telemetry",
        "swarms.telemetry.otel",
        "swarms.utils.generate_id",
        "swarms.utils.litellm_wrapper",
        "swarms.utils.litellm_tokenizer",
        "loguru",
    ]
    modules = {name: ModuleType(name) for name in names}
    modules["swarms.structs.agent"].Agent = OfflineAgent
    modules["swarms.utils.get_cpu_cores"].get_cpu_cores = lambda: 2
    modules["swarms.utils.loguru_logger"].initialize_logger = (
        lambda **kwargs: Mock()
    )
    telemetry = modules["swarms.telemetry.otel"]
    telemetry.ContextThreadPoolExecutor = ThreadPoolExecutor
    telemetry.capture_init = Mock()
    telemetry.trace_run = identity_trace
    modules["swarms.utils.generate_id"].generate_id = (
        lambda *args: uuid.uuid4().hex
    )
    modules["swarms.utils.litellm_wrapper"].empty_usage = dict
    modules["swarms.utils.litellm_tokenizer"].count_tokens = Mock(
        return_value=0
    )
    modules["loguru"].logger = Mock()
    graph = ModuleType("offline_graph_workflow")
    with patch.dict(sys.modules, modules):
        path = ROOT / "swarms/structs/conversation.py"
        exec(  # noqa: S102 - actual local Conversation module
            compile(
                path.read_text(encoding="utf-8"), str(path), "exec"
            ),
            modules["swarms.structs.conversation"].__dict__,
        )
        path = ROOT / "swarms/structs/graph_workflow.py"
        exec(  # noqa: S102 - actual local GraphWorkflow module
            compile(
                path.read_text(encoding="utf-8"), str(path), "exec"
            ),
            graph.__dict__,
        )
    return graph


def run_example(graph):
    swarms = ModuleType("swarms")
    swarms.Agent = OfflineAgent
    for name in ["GraphWorkflow", "Node", "NodeType", "Edge"]:
        setattr(swarms, name, getattr(graph, name))
    namespace = {"__name__": "__main__"}
    output = io.StringIO()
    with patch.dict(sys.modules, {"swarms": swarms}), redirect_stdout(
        output
    ):
        exec(  # noqa: S102 - actual local example
            compile(
                SOURCE.read_text(encoding="utf-8"),
                str(SOURCE),
                "exec",
            ),
            namespace,
        )
    return namespace, output.getvalue()


class BasicGraphTaskNodeTests(unittest.TestCase):
    def setUp(self):
        network = patch.object(
            socket,
            "socket",
            side_effect=AssertionError("Network access is forbidden"),
        )
        network.start()
        self.addCleanup(network.stop)
        self.graph = load_real_graph()

    def test_complete_example_runs_supported_nodes(self):
        namespace, output = run_example(self.graph)
        self.assertEqual(
            namespace["results"],
            {
                "Agent1": "Offline output from Agent1",
                "Agent2": "Offline output from Agent2",
                "Summary": "Offline output from Summary",
            },
        )
        self.assertIn("Execution results:", output)

    def test_summary_receives_both_parent_results(self):
        namespace, _ = run_example(self.graph)
        workflow = namespace["wf_graph"]
        summary = workflow.nodes["Summary"].agent
        self.assertEqual(len(summary.calls), 1)
        _, messages = summary.calls[0]
        context = "\n".join(
            message["content"] for message in messages
        )
        self.assertIn("Offline output from Agent1", context)
        self.assertIn("Offline output from Agent2", context)
        for name in ["Agent1", "Agent2"]:
            self.assertIn(
                "multi-agent", workflow.nodes[name].agent.calls[0][0]
            )
        self.assertEqual(
            set(workflow.entry_points), {"Agent1", "Agent2"}
        )
        self.assertEqual(workflow.end_points, ["Summary"])

    def test_text_visualization_avoids_graphviz_dependency(self):
        with patch.object(
            self.graph.GraphWorkflow,
            "visualize",
            side_effect=AssertionError("Graphviz must not be used"),
        ):
            _, output = run_example(self.graph)
        self.assertIn("Nodes: 3, Edges: 2", output)
        self.assertIn("Agent1", output)
        self.assertIn("Summary", output)


if __name__ == "__main__":
    unittest.main()
