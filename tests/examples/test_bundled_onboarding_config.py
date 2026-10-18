"""Check bundled onboarding through the actual YAML factory."""

import ast
import copy
import inspect
import io
import os
import socket
import sys
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import ClassVar
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "examples/single_agent/getting_started/onboarding"
    / "onboard-basic.py"
)
CONFIG = SOURCE.with_name("agents.yaml")


def actual_agent_signature():
    """Read the real signature without importing or running Agent."""
    path = ROOT / "swarms/structs/agent.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    agent = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Agent"
    )
    constructor = copy.deepcopy(
        next(
            node
            for node in agent.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "__init__"
        )
    )
    constructor.name = "agent_signature"
    constructor.body = [ast.Pass()]
    constructor.returns = None
    constructor.decorator_list = []
    for node in ast.walk(constructor.args):
        if isinstance(node, ast.arg):
            node.annotation = None
    namespace = {}
    constants = ast.parse(
        (ROOT / "swarms/structs/autonomous_loop_utils.py").read_text(
            encoding="utf-8"
        )
    )
    names = {
        "MAX_PLANNING_ATTEMPTS",
        "MAX_SUBTASK_ITERATIONS",
        "MAX_SUBTASK_LOOPS",
    }
    for node in constants.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id in names
                ):
                    namespace[target.id] = ast.literal_eval(
                        node.value
                    )
    exec(  # noqa: S102 - build only the signature function
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[constructor], type_ignores=[])
            ),
            str(path),
            "exec",
        ),
        namespace,
    )
    return inspect.signature(namespace["agent_signature"])


class RecordingAgent:
    instances: ClassVar[list] = []
    signature = None

    def __init__(self, **kwargs):
        self.signature.bind(self, **kwargs)
        self.kwargs = kwargs
        self.agent_name = kwargs["agent_name"]
        self.instances.append(self)

    def run(self, *args, **kwargs):
        raise AssertionError("Onboarding must not run an agent")


class OfflineRouter:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def run(self, *args, **kwargs):
        raise AssertionError("Onboarding must not run the swarm")


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextmanager
def real_yaml_factory():
    modules = {
        name: ModuleType(name)
        for name in [
            "swarms",
            "swarms.structs",
            "swarms.utils",
            "swarms.structs.agent",
            "swarms.structs.swarm_router",
            "swarms.utils.types",
            "swarms.utils.loguru_logger",
            "dotenv",
            "loguru",
        ]
    }
    modules["swarms.structs.agent"].Agent = RecordingAgent
    modules["swarms.structs.swarm_router"].SwarmRouter = OfflineRouter
    modules["swarms.utils.types"].ReturnTypes = str
    modules["swarms.utils.loguru_logger"].initialize_logger = (
        lambda **kwargs: Mock()
    )
    modules["dotenv"].load_dotenv = Mock(return_value=False)
    modules["loguru"].logger = Mock()
    namespace = {"__name__": "offline_yaml_factory"}
    with patch.dict(sys.modules, modules):
        path = ROOT / "swarms/agents/create_agents_from_yaml.py"
        exec(  # noqa: S102 - actual local YAML factory
            compile(
                path.read_text(encoding="utf-8"), str(path), "exec"
            ),
            namespace,
        )
        factory = namespace["create_agents_from_yaml"]
        modules["swarms"].create_agents_from_yaml = factory
        yield factory


def run_onboarding(cwd):
    namespace = {"__name__": "__main__", "__file__": str(SOURCE)}
    with (
        real_yaml_factory(),
        working_directory(cwd),
        redirect_stdout(io.StringIO()),
    ):
        exec(  # noqa: S102 - actual local onboarding script
            compile(
                SOURCE.read_text(encoding="utf-8"),
                str(SOURCE),
                "exec",
            ),
            namespace,
        )
    return namespace


class BundledOnboardingTests(unittest.TestCase):
    def setUp(self):
        RecordingAgent.signature = actual_agent_signature()
        RecordingAgent.instances = []
        network = patch.object(
            socket,
            "socket",
            side_effect=AssertionError("Network access is forbidden"),
        )
        network.start()
        self.addCleanup(network.stop)

    def assert_bundled_agents(self, agents):
        self.assertIsInstance(agents, list)
        self.assertEqual(
            [agent.agent_name for agent in agents],
            ["Financial-Analysis-Agent", "Stock-Analysis-Agent"],
        )
        self.assertEqual(
            [
                (
                    agent.kwargs.get("model_name"),
                    agent.kwargs.get("temperature"),
                    agent.kwargs.get("max_tokens"),
                )
                for agent in agents
            ],
            [("gpt-4", 0.1, 2000), ("gpt-4", 0.2, 1500)],
        )
        self.assertTrue(
            all("model" not in agent.kwargs for agent in agents)
        )

    def test_bundled_yaml_passes_flat_model_parameters(self):
        with real_yaml_factory() as factory:
            agents = factory(
                yaml_file=str(CONFIG), return_type="agents"
            )
        self.assert_bundled_agents(agents)

    def test_onboarding_runs_from_repository_root(self):
        namespace = run_onboarding(ROOT)
        self.assertTrue(
            "task_results" in namespace, "Bundled YAML was not loaded"
        )
        self.assert_bundled_agents(namespace["task_results"])

    def test_unrelated_cwd_configuration_is_not_selected(self):
        with TemporaryDirectory() as directory:
            cwd = Path(directory)
            other = cwd / "agents.yaml"
            content = (
                "agents:\n"
                "  - agent_name: Unrelated-Fixture\n"
                "    system_prompt: Unrelated task\n"
                "    model_name: fixture-model\n"
            )
            other.write_text(content, encoding="utf-8")
            namespace = run_onboarding(cwd)
            self.assertTrue(
                "task_results" in namespace,
                "Bundled YAML was not loaded",
            )
            self.assert_bundled_agents(namespace["task_results"])
            self.assertEqual(
                other.read_text(encoding="utf-8"), content
            )

    def test_model_options_are_explicit_real_agent_parameters(self):
        for name in ["model_name", "temperature", "max_tokens"]:
            with self.subTest(parameter=name):
                self.assertEqual(
                    RecordingAgent.signature.parameters[name].kind,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )


if __name__ == "__main__":
    unittest.main()
