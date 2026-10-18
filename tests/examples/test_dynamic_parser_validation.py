import ast
import json
import unittest
from contextlib import suppress
from pathlib import Path
from typing import Any, Optional, Union
from unittest.mock import Mock, patch

from pydantic import (
    BaseModel,
    Field,
    RootModel,
    ValidationError,
    create_model,
)

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples"
EXAMPLE = EXAMPLE_DIR / "multi_agent/asb/auto_agent.py"


class ConstrainedModel(BaseModel):
    required: str
    quantity: int = Field(gt=0)
    label: str = Field(min_length=3)


class DefaultModel(BaseModel):
    required: str
    count: int = Field(default=7, gt=0)


class TestDynamicParserValidation(unittest.TestCase):
    def setUp(self) -> None:
        wanted = {
            "DynamicParser",
            "Thoughts",
            "Command",
            "AgentResponse",
            "parse_and_execute_command",
        }
        nodes = [
            node
            for node in ast.parse(EXAMPLE.read_text()).body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
            and node.name in wanted
        ]
        self.execute = Mock(return_value="offline command")
        self.namespace = {
            "__name__": "offline_dynamic_parser",
            "json": json,
            "suppress": suppress,
            "Any": Any,
            "Dict": dict,
            "Optional": Optional,
            "Union": Union,
            "Type": type,
            "BaseModel": BaseModel,
            "Field": Field,
            "ValidationError": ValidationError,
            "create_model": create_model,
            "execute_command": self.execute,
        }
        exec(  # noqa: S102 - Load only parser/schema definitions offline.
            compile(
                ast.Module(body=nodes, type_ignores=[]),
                str(EXAMPLE),
                "exec",
            ),
            self.namespace,
        )
        self.parser = self.namespace["DynamicParser"]

    def parse(
        self,
        data: str | dict,
        model: type[BaseModel] = ConstrainedModel,
    ) -> BaseModel | None:
        with patch("socket.socket", side_effect=AssertionError):
            return self.parser.parse(data, model)

    def test_partial_model_keeps_numeric_constraint(self) -> None:
        self.assertIsNone(self.parse({"quantity": -1}))

    def test_partial_model_keeps_string_constraint(self) -> None:
        self.assertIsNone(self.parse({"label": "x"}))

    def test_null_json_returns_none(self) -> None:
        self.assertIsNone(self.parse("null"))

    def test_numeric_json_returns_none(self) -> None:
        self.assertIsNone(self.parse("123"))

    def test_boolean_json_returns_none(self) -> None:
        self.assertIsNone(self.parse("true"))

    def test_valid_partial_model_still_parses(self) -> None:
        result = self.parse({"quantity": 2})
        self.assertEqual(result.quantity, 2)
        self.assertFalse(hasattr(result, "required"))

    def test_valid_full_model_is_unchanged(self) -> None:
        result = self.parse(
            {"required": "present", "quantity": 2, "label": "valid"}
        )
        self.assertIsInstance(result, ConstrainedModel)

    def test_valid_root_model_is_checked_before_object_guard(
        self,
    ) -> None:
        result = self.parse("[1, 2]", RootModel[list[int]])
        self.assertEqual(result.root, [1, 2])

    def test_partial_field_keeps_its_original_default(self) -> None:
        partial = self.parser.create_partial_model(
            DefaultModel, {"count": 3}
        )
        self.assertEqual(partial.model_validate({}).count, 7)

    def test_original_field_metadata_is_not_mutated(self) -> None:
        field = ConstrainedModel.model_fields["quantity"]
        before = repr(field)
        self.parser.create_partial_model(
            ConstrainedModel, {"quantity": 2}
        )
        self.assertEqual(repr(field), before)
        self.assertTrue(field.is_required())

    def test_command_wrapper_rejects_null_without_execution(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            self.namespace["parse_and_execute_command"]("null")
        self.execute.assert_not_called()

    def test_valid_partial_command_reaches_offline_executor(
        self,
    ) -> None:
        result = self.namespace["parse_and_execute_command"](
            {"command": {"name": "do_nothing", "args": {}}}
        )
        self.assertEqual(result, "offline command")
        self.execute.assert_called_once_with("do_nothing", {})

    def test_invalid_array_still_returns_none(self) -> None:
        self.assertIsNone(self.parse("[]"))


if __name__ == "__main__":
    unittest.main()
