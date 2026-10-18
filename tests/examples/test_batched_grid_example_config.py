"""Run example configurations with the actual workflow and helpers."""

import ast
import concurrent.futures
import io
import socket
import sys
import traceback
import typing
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples/multi_agent/batched_grid_workflow"


def exec_definitions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and node.name in names
    ]
    exec(  # noqa: S102 - only local repository definitions
        compile(
            ast.Module(body=definitions, type_ignores=[]),
            str(path),
            "exec",
        ),
        namespace,
    )


def identity_trace(*args, **kwargs):
    return lambda function: function


class OfflineAgent:
    def __init__(self, system_prompt, **kwargs):
        self.system_prompt = system_prompt

    def run(self, task):
        return f"{self.system_prompt}: {task}"


def load_real_workflow():
    namespace = {
        "List": list,
        "Optional": typing.Optional,
        "AgentType": object,
        "Any": object,
        "logger": Mock(),
        "concurrent": concurrent,
        "ContextThreadPoolExecutor": ThreadPoolExecutor,
        "max_workers_95_percent": lambda: 2,
        "capture_init": Mock(),
        "trace_run": identity_trace,
        "generate_id": lambda prefix: prefix,
        "traceback": traceback,
    }
    exec_definitions(
        ROOT / "swarms/structs/multi_agent_exec.py",
        {"run_single_agent", "batched_grid_agent_execution"},
        namespace,
    )
    exec_definitions(
        ROOT / "swarms/structs/batched_grid_workflow.py",
        {"BatchedGridWorkflow"},
        namespace,
    )
    return namespace["BatchedGridWorkflow"]


def run_example(filename, main):
    swarms = ModuleType("swarms")
    swarms.Agent = OfflineAgent
    structs = ModuleType("swarms.structs")
    workflow = ModuleType("swarms.structs.batched_grid_workflow")
    workflow.BatchedGridWorkflow = load_real_workflow()
    namespace = {
        "__name__": "__main__" if main else "offline_grid_example"
    }
    path = EXAMPLES / filename
    with (
        patch.dict(
            sys.modules,
            {
                "swarms": swarms,
                "swarms.structs": structs,
                "swarms.structs.batched_grid_workflow": workflow,
            },
        ),
        redirect_stdout(io.StringIO()),
    ):
        exec(  # noqa: S102 - execute only a local example
            compile(
                path.read_text(encoding="utf-8"), str(path), "exec"
            ),
            namespace,
        )
    return namespace


class BatchedGridExampleConfigTests(unittest.TestCase):
    def setUp(self):
        network = patch.object(
            socket,
            "socket",
            side_effect=AssertionError("Network access is forbidden"),
        )
        network.start()
        self.addCleanup(network.stop)

    def test_comprehensive_example_runs_all_configurations(self):
        namespace = run_example(
            "batched_grid_workflow_example.py", True
        )
        result = namespace["research_result"]
        self.assertEqual(len(result), 2)
        self.assertTrue(all(len(loop) == 3 for loop in result))
        self.assertIn("quantum computing", result[0][0])
        self.assertEqual(len(namespace["content_result"]), 3)
        self.assertEqual(len(namespace["education_result"]), 1)

    def test_advanced_example_runs_all_configurations(self):
        namespace = run_example(
            "batched_grid_advanced_example.py", True
        )
        result = namespace["custom_result"]
        self.assertEqual(len(result), 2)
        self.assertTrue(all(len(loop) == 3 for loop in result))
        self.assertIn("artificial intelligence safety", result[0][0])
        self.assertEqual(len(namespace["refinement_result"]), 4)
        self.assertEqual(len(namespace["qa_result"]), 1)

    def test_basic_workflow_preserves_paired_task_order(self):
        namespace = run_example(
            "batched_grid_workflow_example.py", False
        )
        result = namespace["basic_batched_workflow"]()
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0]), 2)
        self.assertIn("robot learning to paint", result[0][0])
        self.assertIn("creative industries", result[0][1])


if __name__ == "__main__":
    unittest.main()
