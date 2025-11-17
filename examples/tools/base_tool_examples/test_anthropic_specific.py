#!/usr/bin/env python3
"""
Test script specifically for Anthropic function call execution based on the
tool_schema.py output shown by the user.
"""

from swarms.tools.base_tool import BaseTool
from pydantic import BaseModel
import json


def get_current_weather(location: str, unit: str = "celsius") -> dict:
    """Get the current weather in a given location"""
    return {
        "location": location,
        "temperature": "22" if unit == "celsius" else "72",
        "unit": unit,
        "condition": "sunny",
        "description": f"The weather in {location} is sunny with a temperature of {'22°C' if unit == 'celsius' else '72°F'}",
    }


# Simulate the actual response structure from the tool_schema.py output
class ChatCompletionMessageToolCall(BaseModel):
    index: int
    function: "Function"
    id: str
    type: str


class Function(BaseModel):
    arguments: str
    name: str


def test_litellm_anthropic_response():
    """Test the exact response structure from the tool_schema.py output"""
    print("=== Testing LiteLLM Anthropic Response Structure ===")

    tool = BaseTool(tools=[get_current_weather], verbose=True)

    # Create the exact structure from your output
    tool_call = ChatCompletionMessageToolCall(
        index=1,
        function=Function(
            arguments='{"location": "Boston", "unit": "fahrenheit"}',
            name="get_current_weather",
        ),
        id="toolu_019vcXLipoYHzd1e1HUYSSaa",
        type="function",
    )

    # Test with single BaseModel object
    print("Testing single ChatCompletionMessageToolCall:")
    try:
        results = tool.execute_function_calls_from_api_response(
            tool_call
        )
        print("Results:")
        for result in results:
            print(f"  {result}")
        print()
    except Exception as e:
        print(f"Error: {e}")
        print()

    # Test with list of BaseModel objects (as would come from tool_calls)
    print("Testing list of ChatCompletionMessageToolCall:")
    try:
        results = tool.execute_function_calls_from_api_response(
            [tool_call]
        )
        print("Results:")
        for result in results:
            print(f"  {result}")
        print()
    except Exception as e:
        print(f"Error: {e}")
        print()


def test_format_detection():
    """Test format detection for the specific structure"""
    print("=== Testing Format Detection ===")

    tool = BaseTool()

    # Test the BaseModel from your output
    tool_call = ChatCompletionMessageToolCall(
        index=1,
        function=Function(
            arguments='{"location": "Boston", "unit": "fahrenheit"}',
            name="get_current_weather",
        ),
        id="toolu_019vcXLipoYHzd1e1HUYSSaa",
        type="function",
    )

    detected_format = tool.detect_api_response_format(tool_call)
    print(
        f"Detected format for ChatCompletionMessageToolCall: {detected_format}"
    )

    # Test the converted dictionary
    tool_call_dict = tool_call.model_dump()
    print(
        f"Tool call as dict: {json.dumps(tool_call_dict, indent=2)}"
    )

    detected_format_dict = tool.detect_api_response_format(
        tool_call_dict
    )
    print(
        f"Detected format for converted dict: {detected_format_dict}"
    )
    print()


def test_manual_conversion():
    """Test manual conversion and execution"""
    print("=== Testing Manual Conversion ===")

    tool = BaseTool(tools=[get_current_weather], verbose=True)

    # Create the BaseModel
    tool_call = ChatCompletionMessageToolCall(
        index=1,
        function=Function(
            arguments='{"location": "Boston", "unit": "fahrenheit"}',
            name="get_current_weather",
        ),
        id="toolu_019vcXLipoYHzd1e1HUYSSaa",
        type="function",
    )

    # Manually convert to dict
    tool_call_dict = tool_call.model_dump()
    print(
        f"Converted to dict: {json.dumps(tool_call_dict, indent=2)}"
    )

    # Try to execute
    try:
        results = tool.execute_function_calls_from_api_response(
            tool_call_dict
        )
        print("Manual conversion results:")
        for result in results:
            print(f"  {result}")
        print()
    except Exception as e:
        print(f"Error with manual conversion: {e}")
        print()


if __name__ == "__main__":
    print("Testing Anthropic-Specific Function Call Execution\n")

    test_format_detection()
    test_manual_conversion()
    test_litellm_anthropic_response()

    print("=== All Anthropic Tests Complete ===")
