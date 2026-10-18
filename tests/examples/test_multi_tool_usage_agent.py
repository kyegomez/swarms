"""Offline regressions for the multi-tool example's execution loop."""

import contextlib
import io
import json
import os
import runpy
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "examples/single_agent/capabilities/tools/multi_tool_usage_agent.py"
)


class ToolAgentProgressTests(unittest.TestCase):
    def setUp(self) -> None:
        swarms = types.ModuleType("swarms")
        swarms.Agent = Mock()
        swarms.Agent.return_value.run.side_effect = [
            json.dumps({"plan": {"steps": []}}),
            json.dumps({"summary": "offline import"}),
        ]
        with (
            patch.dict(sys.modules, {"swarms": swarms}),
            patch.object(os, "getenv", return_value=None),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.tool_agent_type = runpy.run_path(str(EXAMPLE))[
                "ToolAgent"
            ]

    def assert_plan_runs_once(self, actions: tuple[str, ...]) -> None:
        calls = []

        def record(value: int) -> int:
            """Record a tool side effect and bound a broken loop."""
            calls.append(value)
            if calls.count(value) > 1:
                raise RuntimeError("completed tool step was repeated")
            return value * 10

        steps = [
            {
                "step_id": index,
                "function": "record",
                "parameters": {"value": index},
                "purpose": f"Record step {index}",
            }
            for index in range(1, len(actions) + 1)
        ]
        responses = [{"plan": {"steps": steps}}]
        responses.extend(
            {"next_action": {"type": action}} for action in actions
        )
        responses.append({"summary": "all steps completed"})
        agent = self.tool_agent_type([record], "unused-offline-key")
        agent.agent.run = Mock(
            side_effect=[
                json.dumps(response) for response in responses
            ]
        )
        with contextlib.redirect_stdout(io.StringIO()):
            result = agent.run("Record every planned step")

        self.assertEqual(calls, list(range(1, len(actions) + 1)))
        self.assertTrue(result["final_result"]["success"], result)
        self.assertEqual(len(result["steps"]), len(actions))
        self.assertEqual(
            result["final_result"]["results"],
            {index: index * 10 for index in calls},
        )
        self.assertEqual(
            result["final_result"]["analysis"], responses[-1]
        )
        self.assertEqual(agent.agent.run.call_count, len(actions) + 2)

    def test_early_complete_advances_to_remaining_steps(self) -> None:
        for actions in (
            ("complete", "complete"),
            ("complete", "complete", "complete"),
            ("continue", "complete", "complete"),
        ):
            with self.subTest(actions=actions):
                self.assert_plan_runs_once(actions)

    def test_continue_finishes_each_step_once(self) -> None:
        self.assert_plan_runs_once(("continue", "continue"))

    def test_complete_on_final_step_generates_summary(self) -> None:
        self.assert_plan_runs_once(("complete",))
        self.assert_plan_runs_once(("continue", "complete"))


if __name__ == "__main__":
    unittest.main()
