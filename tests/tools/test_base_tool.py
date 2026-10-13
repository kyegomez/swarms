import json
from typing import Optional

import pytest
from pydantic import BaseModel

from swarms.tools.base_tool import (
    BaseTool,
    ToolDocumentationError,
    ToolTypeHintError,
)


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

    result = tool.func_to_dict(sample_function)

    assert result["type"] == "function"
    assert result["function"]["name"] == "sample_function"
    assert "parameters" in result["function"]
    print("func_to_dict test passed")


def test_base_model_to_dict():
    print("Testing base_model_to_dict")
    tool = BaseTool()

    result = tool.base_model_to_dict(TestModel)

    assert result["type"] == "function"
    assert "name" in result["function"]["parameters"]["properties"]
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
    assert json.loads(result)["results"] == {"sample_function": "3"}
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


def test_multiple_functions_to_dict_on_one_cpu(monkeypatch):
    monkeypatch.setattr("os.cpu_count", lambda: 1)

    result = BaseTool().multiple_functions_to_dict([sample_function])

    assert [r["function"]["name"] for r in result] == [
        "sample_function"
    ]


class LineItem(BaseModel):
    sku: str
    qty: int


def place_order(items: list[LineItem], note: str = "") -> str:
    """Place an order."""
    return note


def test_func_to_dict_resolves_nested_model_refs():
    params = BaseTool().func_to_dict(place_order)["function"][
        "parameters"
    ]

    items = params["properties"]["items"]
    assert items["items"] == {"$ref": "#/$defs/LineItem"}
    assert "$defs" not in items
    assert params["$defs"]["LineItem"]["required"] == ["sku", "qty"]


def add_numbers(a: int, b: int) -> int:
    """Add two numbers.

    Args:
        a (int): First number.
        b (int): Second number.
    """
    return a + b


def get_weather(location: str, unit: str = "celsius") -> dict:
    """Get the weather for a location."""
    return {"location": location, "unit": unit}


def test_multi_base_models_to_dict():
    class ProductModel(BaseModel):
        title: str

    result = BaseTool().multi_base_models_to_dict(
        [TestModel, ProductModel]
    )

    assert [r["function"]["name"] for r in result] == [
        "TestModel",
        "ProductModel",
    ]


def test_convert_tool_into_openai_schema():
    tool = BaseTool(tools=[add_numbers, get_weather])

    result = tool.convert_tool_into_openai_schema()

    assert result["type"] == "function"
    assert [f["name"] for f in result["functions"]] == [
        "add_numbers",
        "get_weather",
    ]


def test_tools_must_have_docs_and_type_hints():
    def undocumented(x: int) -> int:
        return x

    def untyped(x):
        """No type hints."""
        return x

    tool = BaseTool()

    assert tool.check_func_if_have_docs(add_numbers) is True
    assert tool.check_func_if_have_type_hints(add_numbers) is True
    with pytest.raises(ToolDocumentationError):
        tool.check_func_if_have_docs(undocumented)
    with pytest.raises(ToolTypeHintError):
        tool.check_func_if_have_type_hints(untyped)


def test_find_function_name():
    tool = BaseTool(tools=[add_numbers, get_weather])

    assert tool.find_function_name("get_weather") is get_weather
    assert tool.find_function_name("missing") is None


def test_function_to_dict():
    result = BaseTool().function_to_dict(add_numbers)

    assert result["function"]["name"] == "add_numbers"


class Function(BaseModel):
    name: str
    arguments: str


class ChatCompletionMessageToolCall(BaseModel):
    index: int
    id: str
    type: str
    function: Function


class AnthropicToolCall(BaseModel):
    type: str = "tool_use"
    id: str
    name: str
    input: dict


BOSTON = {"location": "Boston", "unit": "fahrenheit"}
WEATHER_CALL = ChatCompletionMessageToolCall(
    index=0,
    id="call_1",
    type="function",
    function=Function(
        name="get_weather", arguments=json.dumps(BOSTON)
    ),
)


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": json.dumps(
                                            BOSTON
                                        ),
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
            id="openai-response",
        ),
        pytest.param(WEATHER_CALL, id="tool-call-model"),
        pytest.param([WEATHER_CALL], id="list-of-tool-call-models"),
        pytest.param(WEATHER_CALL.model_dump(), id="tool-call-dict"),
        pytest.param(
            {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get_weather",
                        "input": BOSTON,
                    }
                ]
            },
            id="anthropic-response",
        ),
        pytest.param(
            AnthropicToolCall(
                id="toolu_1", name="get_weather", input=BOSTON
            ),
            id="anthropic-tool-use-model",
        ),
    ],
)
def test_execute_function_calls_from_each_response_format(response):
    tool = BaseTool(tools=[get_weather])

    results = tool.execute_function_calls_from_api_response(
        response, return_as_string=False
    )

    assert results == [BOSTON]


def test_a_list_of_anthropic_tool_calls_runs_each_tool_once():
    ran = []

    def record(name: str) -> str:
        """Record a call."""
        ran.append(name)
        return name

    BaseTool(tools=[record]).execute_function_calls_from_api_response(
        [
            AnthropicToolCall(
                id="a", name="record", input={"name": "1"}
            ),
            AnthropicToolCall(
                id="b", name="record", input={"name": "2"}
            ),
        ]
    )

    assert ran == ["1", "2"]


class SdkFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = json.dumps(arguments)


class SdkToolCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.type = "function"
        self.function = SdkFunction(name, arguments)


@pytest.mark.parametrize("sequential", [True, False])
def test_tool_call_objects_return_results_in_call_order(sequential):
    calls = [
        SdkToolCall("c1", "add_numbers", {"a": 2, "b": 3}),
        SdkToolCall("c2", "get_weather", {"location": "Paris"}),
    ]
    tool = BaseTool(tools=[add_numbers, get_weather])

    raw = tool.execute_function_calls_from_api_response(
        calls, sequential=sequential, return_as_string=False
    )
    text = tool.execute_function_calls_from_api_response(
        calls, sequential=sequential
    )

    assert raw == [5, {"location": "Paris", "unit": "celsius"}]
    assert text[0] == "Function 'add_numbers' result:\n5"


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
