"""Run the complete swarm example with offline agent doubles."""

import io
import socket
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "examples/multi_agent/social_algorithms_examples"
    / "swarm_intelligence_algorithm_example.py"
)


class OfflineAgent:
    def __init__(self, agent_name, **kwargs):
        self.agent_name = agent_name
        self.calls = []

    def run(self, prompt):
        self.calls.append(prompt)
        return f"Offline response from {self.agent_name}"


class OfflineSocialAlgorithms:
    def __init__(self, agents, social_algorithm, **kwargs):
        self.agents = agents
        self.social_algorithm = social_algorithm

    def run(self, task, algorithm_args=None):
        outputs = self.social_algorithm(
            self.agents, task, **(algorithm_args or {})
        )
        return SimpleNamespace(
            final_outputs=outputs,
            total_steps=sum(
                len(agent.calls) for agent in self.agents
            ),
            execution_time=0.0,
        )


def load_example(name):
    swarms = ModuleType("swarms")
    swarms.Agent = OfflineAgent
    structs = ModuleType("swarms.structs")
    algorithms = ModuleType("swarms.structs.social_algorithms")
    algorithms.SocialAlgorithms = OfflineSocialAlgorithms
    namespace = {"__name__": name}
    output = io.StringIO()
    with (
        patch.dict(
            sys.modules,
            {
                "swarms": swarms,
                "swarms.structs": structs,
                "swarms.structs.social_algorithms": algorithms,
            },
        ),
        redirect_stdout(output),
    ):
        exec(  # noqa: S102 - execute only the local example
            compile(
                SOURCE.read_text(encoding="utf-8"),
                str(SOURCE),
                "exec",
            ),
            namespace,
        )
    return namespace, output.getvalue()


class SwarmIntelligenceSummaryTests(unittest.TestCase):
    def setUp(self):
        network = patch.object(
            socket,
            "socket",
            side_effect=AssertionError("Network access is forbidden"),
        )
        network.start()
        self.addCleanup(network.stop)

    def test_main_completes_and_prints_iteration_count(self):
        namespace, output = load_example("__main__")
        self.assertEqual(
            namespace["result"].final_outputs["total_loops"], 8
        )
        self.assertIn("Total Iterations: 8", output)
        self.assertIn("=== FINAL SWARM SYNTHESIS ===", output)
        self.assertIn("=== SWARM BEHAVIOR PATTERNS ===", output)
        self.assertIn(
            "Offline response from Swarm_Coordinator", output
        )

    def test_zero_loops_still_returns_initial_discoveries(self):
        namespace, _ = load_example("offline_swarm_example")
        result = namespace["social_alg"].run(
            "Offline task", algorithm_args={"max_loops": 0}
        )
        self.assertEqual(result.final_outputs["total_loops"], 0)
        self.assertEqual(result.final_outputs["total_discoveries"], 5)

    def test_configured_loop_count_is_preserved(self):
        namespace, _ = load_example("offline_swarm_example")
        result = namespace["social_alg"].run(
            "Offline task", algorithm_args={"max_loops": 2}
        )
        self.assertEqual(result.final_outputs["total_loops"], 2)
        self.assertEqual(
            result.final_outputs["total_discoveries"], 15
        )


if __name__ == "__main__":
    unittest.main()
