import ast
import json
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional
from unittest.mock import Mock, patch

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples"
EXAMPLE = (
    EXAMPLE_DIR
    / "multi_agent/council/judges/council_judge_evaluation.py"
)


class TestCouncilGsm8kAnswers(unittest.TestCase):
    def setUp(self) -> None:
        tree = ast.parse(EXAMPLE.read_text())
        definition = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "CouncilJudgeEvaluator"
        )
        self.load_dataset = Mock()
        namespace = {
            "Any": Any,
            "Dict": dict,
            "Optional": Optional,
            "Agent": type("OfflineAgent", (), {}),
            "base_agent": Mock(),
            "CouncilAsAJudge": Mock(),
            "Path": Path,
            "json": json,
            "time": time,
            "logger": Mock(),
            "load_dataset": self.load_dataset,
            "DATASET_CONFIGS": {"gsm8k": "main"},
            "tqdm": lambda rows, **kwargs: rows,
        }
        exec(  # noqa: S102 - Load the class with offline dependencies.
            compile(
                ast.Module(body=[definition], type_ignores=[]),
                str(EXAMPLE),
                "exec",
            ),
            namespace,
        )
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.evaluator = namespace["CouncilJudgeEvaluator"](
            output_dir=self.directory.name
        )

    def evaluate(self, reference: str, response: str) -> dict:
        self.load_dataset.return_value = [
            {
                "question": "Solve this synthetic problem.",
                "answer": reference,
            }
        ]
        self.evaluator.council.run.return_value = response
        with patch("socket.socket", side_effect=AssertionError):
            result = self.evaluator.evaluate_dataset(
                "gsm8k", save_results=False
            )
        self.load_dataset.assert_called_once_with(
            "gsm8k", "main", split="test"
        )
        self.evaluator.council.run.assert_called_once()
        return result

    def test_reference_extracts_final_answer_after_delimiter(
        self,
    ) -> None:
        actual = self.evaluator._get_correct_answer(
            {"answer": "Compute six plus six.\n#### 12"}, "gsm8k"
        )
        self.assertEqual(actual, "12")

    def test_reference_normalizes_signed_decimal_and_commas(
        self,
    ) -> None:
        actual = self.evaluator._get_correct_answer(
            {"answer": "Synthetic calculation.\n#### -1,234.5"},
            "gsm8k",
        )
        self.assertEqual(actual, "-1234.5")

    def test_positive_final_answer_is_scored_correctly(self) -> None:
        result = self.evaluate("Six plus six.\n#### 12", "Answer: 12")
        self.assertEqual(result["correct_answers"], 1)
        self.assertEqual(result["accuracy"], 1.0)

    def test_negative_decimal_final_answer_is_scored_correctly(
        self,
    ) -> None:
        result = self.evaluate(
            "Calculation.\n#### -2.5", "Answer: -2.5"
        )
        self.assertEqual(result["correct_answers"], 1)

    def test_formatted_thousands_are_scored_correctly(self) -> None:
        result = self.evaluate(
            "Calculation.\n#### 1,234.5", "The answer is 1,234.5"
        )
        self.assertEqual(result["correct_answers"], 1)

    def test_wrong_explicit_answer_is_not_marked_correct(
        self,
    ) -> None:
        result = self.evaluate("Calculation.\n#### 12", "Answer: 13")
        self.assertEqual(result["correct_answers"], 0)
        self.assertEqual(result["accuracy"], 0.0)

    def test_official_answer_delimiter_is_scored_correctly(
        self,
    ) -> None:
        result = self.evaluate(
            "Calculation.\n#### 12", "Reasoning.\n#### 12"
        )
        self.assertEqual(result["correct_answers"], 1)

    def test_plain_reference_answer_remains_supported(self) -> None:
        result = self.evaluate("12", "The answer is 12")
        self.assertEqual(result["correct_answers"], 1)

    def test_other_dataset_reference_is_unchanged(self) -> None:
        self.assertEqual(
            self.evaluator._get_correct_answer(
                {"answer": "2"}, "winogrande"
            ),
            "2",
        )


if __name__ == "__main__":
    unittest.main()
