"""Offline checks for the long-form example's Agent configuration."""

import runpy
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "examples/guides/generation_length_blog/longform_generator.py"
)


class LongFormAgentOptionsTests(unittest.TestCase):
    def setUp(self) -> None:
        swarms = ModuleType("swarms")
        self.agent = Mock(
            return_value=Mock(run=Mock(return_value="two words"))
        )
        swarms.Agent = self.agent
        tokenizer = ModuleType("swarms.utils.litellm_tokenizer")
        self.count_tokens = Mock(return_value=7)
        tokenizer.count_tokens = self.count_tokens
        with patch.dict(
            sys.modules,
            {
                "swarms": swarms,
                "swarms.utils.litellm_tokenizer": tokenizer,
            },
        ):
            namespace = runpy.run_path(str(EXAMPLE))
        self.generator = namespace["LongFormGenerator"](
            model="anthropic/test-model"
        )

    def assert_agent_options(self, names: list[str]) -> None:
        """Every creation path must supply the supported Agent keywords."""
        self.assertEqual(self.agent.call_count, len(names))
        for call, name in zip(self.agent.call_args_list, names):
            self.assertEqual(call.kwargs.get("agent_name"), name)
            self.assertEqual(
                call.kwargs.get("model_name"), "anthropic/test-model"
            )
            self.assertNotIn("name", call.kwargs)
            self.assertNotIn("model", call.kwargs)

    def test_initial_generation_uses_selected_model_and_name(
        self,
    ) -> None:
        result = self.generator.generate_with_continuation(
            "topic", {"min_words": 2}
        )
        self.assertEqual(result, "two words")
        self.assert_agent_options(["LongForm Content Generator"])

    def test_continuation_uses_selected_model_and_name(self) -> None:
        with patch("time.sleep"):
            result = self.generator.generate_with_continuation(
                "topic", {"min_words": 4}, max_attempts=1
            )
        self.assertEqual(result, "two words\n\ntwo words")
        self.assert_agent_options(
            [
                "LongForm Content Generator",
                "Content Continuation Agent",
            ]
        )

    def test_sections_use_selected_model_and_distinct_names(
        self,
    ) -> None:
        with patch("time.sleep"):
            result = self.generator.generate_sectioned_content(
                "topic", [{"title": "Overview"}, {"title": "Details"}]
            )
        self.assertEqual(result["Overview"], "two words")
        self.assertEqual(result["Details"], "two words")
        self.assert_agent_options(
            [
                "Section Generator - Overview",
                "Section Generator - Details",
            ]
        )

    def test_tokenizer_keeps_its_model_keyword(self) -> None:
        self.assertEqual(self.generator.estimate_tokens("text"), 7)
        self.count_tokens.assert_called_once_with(
            text="text", model="anthropic/test-model"
        )


if __name__ == "__main__":
    unittest.main()
