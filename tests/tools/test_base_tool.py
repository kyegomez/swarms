import json
from typing import Optional

from pydantic import BaseModel

from swarms.tools.base_tool import BaseTool


class TestModel(BaseModel):
    name: str
    age: int
    email: Optional[str] = None


def sample_function(x: int, y: int) -> int:
    """Test function for addition."""
    return x + y


def test_func_to_dict():
    print("Testing func_to_dict")
    tool = BaseTool()

    result = tool.func_to_dict(
        function=sample_function,
        name="sample_function",
        description="Test function",
    )

    assert result["type"] == "function"
    assert result["function"]["name"] == "sample_function"
    assert "parameters" in result["function"]
    print("func_to_dict test passed")


def test_base_model_to_dict():
    print("Testing base_model_to_dict")
    tool = BaseTool()

    result = tool.base_model_to_dict(TestModel)

    assert "type" in result
    assert "properties" in result["properties"]
    assert "name" in result["properties"]["properties"]
    print("base_model_to_dict test passed")


def test_detect_tool_input_type():
    print("Testing detect_tool_input_type")
    tool = BaseTool()

    model = TestModel(name="Test", age=25)
    assert tool.detect_tool_input_type(model) == "Pydantic"

    dict_input = {"key": "value"}
    assert tool.detect_tool_input_type(dict_input) == "Dictionary"

    assert tool.detect_tool_input_type(sample_function) == "Function"
    print("detect_tool_input_type test passed")


def test_execute_tool_by_name():
    print("Testing execute_tool_by_name")
    tool = BaseTool(
        function_map={"sample_function": sample_function},
        verbose=True,
    )

    response = json.dumps(
        {"name": "sample_function", "parameters": {"x": 1, "y": 2}}
    )

    result = tool.execute_tool_by_name("sample_function", response)
    assert result == 3
    print("execute_tool_by_name test passed")


def test_check_str_for_functions_valid():
    print("Testing check_str_for_functions_valid")
    tool = BaseTool(function_map={"test_func": lambda x: x})

    valid_json = json.dumps(
        {"type": "function", "function": {"name": "test_func"}}
    )

    assert tool.check_str_for_functions_valid(valid_json) is True

    invalid_json = json.dumps({"type": "invalid"})
    assert tool.check_str_for_functions_valid(invalid_json) is False
    print("check_str_for_functions_valid test passed")


def test_convert_funcs_into_tools():
    print("Testing convert_funcs_into_tools")
    tool = BaseTool(tools=[sample_function])

    tool.convert_funcs_into_tools()
    assert "sample_function" in tool.function_map
    assert callable(tool.function_map["sample_function"])
    print("convert_funcs_into_tools test passed")


def run_all_tests():
    print("Starting all tests")

    tests = [
        test_func_to_dict,
        test_base_model_to_dict,
        test_detect_tool_input_type,
        test_execute_tool_by_name,
        test_check_str_for_functions_valid,
        test_convert_funcs_into_tools,
    ]

    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"Test {test.__name__} failed: {str(e)}")
        except Exception as e:
            print(f"Unexpected error in {test.__name__}: {str(e)}")

    print("All tests completed")


if __name__ == "__main__":
    run_all_tests()
