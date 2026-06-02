"""Regression tests for robust function->OpenAI-schema generation.

Covers three defects in the schema generation helpers:
  (A) func_to_str.function_to_str raised KeyError('type') for parameters whose
      JSON schema uses 'anyOf'/'oneOf' (e.g. Optional/Union) instead of a
      top-level 'type'.
  (B) get_openai_function_schema_from_func raised TypeError for functions with
      unannotated *args/**kwargs because variadic parameters were counted as
      required and as missing annotations.
  (C) get_required_params / get_default_values compared defaults with ==/!=
      against inspect.Signature.empty, which misclassifies (and can crash) for
      defaults whose __eq__ returns a non-bool value.
"""

from typing import Optional

from swarms.tools.func_to_str import function_to_str
from swarms.tools.py_func_to_openai_func_str import (
    get_openai_function_schema_from_func,
)


def test_optional_param_function_to_str():
    """function_to_str must not raise on Optional/Union (anyOf) parameters."""

    def func_with_optional(a: Optional[int] = None) -> None:
        """A function with an optional parameter."""

    schema = get_openai_function_schema_from_func(func_with_optional)
    properties = schema["function"]["parameters"]["properties"]

    # The Optional[int] schema uses anyOf, not a top-level 'type'.
    assert "anyOf" in properties["a"]
    assert "type" not in properties["a"]

    # This previously raised KeyError('type').
    result = function_to_str(schema["function"])
    assert "func_with_optional" in result
    # The anyOf member types should be reflected in the rendered string.
    assert "integer" in result


def test_variadic_args_kwargs_schema():
    """Unannotated *args/**kwargs must not be treated as required params."""

    def func_with_variadics(x: int, *args, **kwargs) -> None:
        """A function with variadic parameters."""

    # This previously raised TypeError about missing annotations for
    # 'args' and 'kwargs'.
    schema = get_openai_function_schema_from_func(func_with_variadics)
    parameters = schema["function"]["parameters"]

    # Only the real, annotated parameter should be required.
    assert parameters["required"] == ["x"]
    # Variadics must not leak into the emitted properties.
    assert "args" not in parameters["properties"]
    assert "kwargs" not in parameters["properties"]
    assert "x" in parameters["properties"]


def test_default_with_non_bool_eq():
    """Defaults whose __eq__ returns a non-bool must be classified correctly."""

    class NonBoolEq:
        """A value whose __eq__ returns a non-bool (truthy) value."""

        def __eq__(self, other):
            return [1, 2, 3]

        def __hash__(self):
            return 0

    sentinel = NonBoolEq()

    def func_with_weird_default(x: int, y: int = sentinel) -> None:
        """A function with a default that has a non-bool __eq__."""

    # This previously misclassified 'y' as required (its default compared
    # truthy against inspect.Signature.empty) and raised TypeError.
    schema = get_openai_function_schema_from_func(func_with_weird_default)
    parameters = schema["function"]["parameters"]

    # Only 'x' (no default) should be required; 'y' has a default.
    assert parameters["required"] == ["x"]


if __name__ == "__main__":
    test_optional_param_function_to_str()
    test_variadic_args_kwargs_schema()
    test_default_with_non_bool_eq()
    print("All schema generation robustness tests passed.")
