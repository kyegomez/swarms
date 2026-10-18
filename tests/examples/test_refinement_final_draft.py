"""Check that final integration receives the reviewed draft."""

import ast
import socket
import unittest
from pathlib import Path
from unittest.mock import patch

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "examples/multi_agent/social_algorithms_examples"
    / "iterative_refinement_algorithm_example.py"
)


def load_refinement():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    ]
    namespace = {}
    exec(  # noqa: S102 - only local example functions
        compile(
            ast.Module(body=functions, type_ignores=[]),
            str(SOURCE),
            "exec",
        ),
        namespace,
    )
    return namespace["iterative_refinement_algorithm"]


class RecordingAgent:
    def __init__(self, name, responses):
        self.agent_name = name
        self.responses = iter(responses)
        self.prompts = []

    def run(self, prompt):
        self.prompts.append(prompt)
        return next(self.responses)


class Coordinator:
    agent_name = "Coordinator"

    def __init__(self):
        self.prompts = []

    def run(self, prompt):
        self.prompts.append(prompt)
        return (
            "FINAL" if "final integrated version" in prompt else "OK"
        )


class RefinementFinalDraftTests(unittest.TestCase):
    def setUp(self):
        network = patch.object(
            socket,
            "socket",
            side_effect=AssertionError("Network access is forbidden"),
        )
        network.start()
        self.addCleanup(network.stop)
        self.refine = load_refinement()

    def run_refinement(self, scores):
        count = len(scores)
        creator = RecordingAgent("Creator", ["INITIAL_DRAFT"])
        technical = RecordingAgent(
            "Technical", [f"TECHNICAL_{i}" for i in range(count)]
        )
        style = RecordingAgent(
            "Style", [f"REVIEWED_STYLE_{i}" for i in range(count)]
        )
        quality = RecordingAgent(
            "Quality",
            [
                *[f"Overall quality: {score}" for score in scores],
                "OK",
            ],
        )
        coordinator = Coordinator()
        result = self.refine(
            [creator, technical, style, quality, coordinator],
            "Offline task",
            max_loops=count + 1,
            quality_threshold=8,
        )
        final_prompt = next(
            prompt
            for prompt in coordinator.prompts
            if "Create the final integrated version" in prompt
        )
        return result, final_prompt, style, quality

    def test_first_accepted_draft_is_used_for_final_integration(self):
        result, prompt, style, quality = self.run_refinement([9])
        self.assertIn("Latest Version: REVIEWED_STYLE_0", prompt)
        self.assertNotIn("Latest Version: INITIAL_DRAFT", prompt)
        self.assertEqual(len(style.prompts), 1)
        self.assertIn("REVIEWED_STYLE_0", quality.prompts[0])
        self.assertEqual(result["final_version"], "FINAL")

    def test_later_accepted_draft_replaces_previous_version(self):
        result, prompt, style, _ = self.run_refinement([6, 9])
        self.assertIn("Latest Version: REVIEWED_STYLE_1", prompt)
        self.assertNotIn("Latest Version: REVIEWED_STYLE_0", prompt)
        self.assertEqual(len(style.prompts), 2)
        self.assertEqual(result["final_quality_score"], 9)

    def test_latest_draft_is_kept_when_budget_is_exhausted(self):
        result, prompt, style, _ = self.run_refinement([5, 6])
        self.assertIn("Latest Version: REVIEWED_STYLE_1", prompt)
        self.assertEqual(len(style.prompts), 2)
        self.assertEqual(result["quality_scores"], [5, 6])


if __name__ == "__main__":
    unittest.main()
