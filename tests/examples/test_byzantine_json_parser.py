import json
import runpy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples"
EXAMPLE = EXAMPLE_DIR / "changelogs/v12_update/byzantine_consensus.py"


class TestByzantineJsonParser(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = Mock()
        with (
            patch.dict(
                sys.modules,
                {"swarms": SimpleNamespace(Agent=self.agent)},
            ),
            patch("socket.socket", side_effect=AssertionError),
        ):
            self.namespace = runpy.run_path(str(EXAMPLE))
        self.parse = self.namespace["_parse_json"]

    def test_brace_inside_strategy_string_is_preserved(self) -> None:
        response = {
            "internal_strategy": "Keep this } in the private notes",
            "value": 1,
            "vote": "stop",
        }
        self.assertEqual(self.parse(json.dumps(response)), response)

    def test_nested_metadata_does_not_truncate_proposal(self) -> None:
        response = {
            "internal_strategy": "Agree",
            "metadata": {"round": 1},
            "value": 1,
            "vote": "stop",
        }
        self.assertEqual(self.parse(json.dumps(response)), response)

    def test_latest_complete_history_object_is_selected(self) -> None:
        previous = {"value": 0, "vote": "continue"}
        current = {
            "internal_strategy": "Keep } and agree",
            "value": 1,
            "vote": "stop",
        }
        history = (
            json.dumps(previous)
            + "\n```json\n"
            + json.dumps(current)
            + "\n```"
        )
        self.assertEqual(self.parse(history), current)

    def test_literal_code_fence_inside_notes_is_unchanged(
        self,
    ) -> None:
        response = {
            "internal_strategy": "Keep the literal ```json marker",
            "value": 1,
            "vote": "stop",
        }
        self.assertEqual(self.parse(json.dumps(response)), response)

    def test_invalid_text_returns_empty_object(self) -> None:
        self.assertEqual(self.parse("No JSON {unfinished"), {})

    def test_incomplete_tail_keeps_previous_complete_object(
        self,
    ) -> None:
        response = {"value": 1, "vote": "stop"}
        text = json.dumps(response) + '\n{"unfinished":'
        self.assertEqual(self.parse(text), response)

    def test_plain_json_still_parses(self) -> None:
        response = {"value": 1, "vote": "stop"}
        self.assertEqual(self.parse(json.dumps(response)), response)

    def test_valid_proposal_reaches_consensus_in_actual_run(
        self,
    ) -> None:
        self.agent.return_value.run.return_value = json.dumps(
            {
                "internal_strategy": "Keep } and agree",
                "value": 1,
                "vote": "stop",
            }
        )
        game = self.namespace["ByzantineConsensus"](
            n_honest=1, max_rounds=1, verbose=False
        )
        with (
            patch.object(
                self.namespace["random"], "randint", return_value=1
            ),
            patch("socket.socket", side_effect=AssertionError),
        ):
            self.assertEqual(game.run("Agree"), "valid_consensus")
        self.agent.return_value.run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
