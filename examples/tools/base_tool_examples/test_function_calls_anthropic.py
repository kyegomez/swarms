#!/usr/bin/env python3
"""
Test script to verify the modified execute_function_calls_from_api_response method
works with both OpenAI and Anthropic function calls, including BaseModel objects.
"""

from swarms.tools.base_tool import BaseTool
from pydantic import BaseModel


# Example functions to test with
def get_current_weather(location: str, unit: str = "celsius") -> dict:
    """Get the current weather in a given location"""
    return {
        "location": location,
        "temperature": "22" if unit == "celsius" else "72",
        "unit": unit,
        "condition": "sunny",
    }


def calculate_sum(a: int, b: int) -> int:
    """Calculate the sum of two numbers"""
    return a + b


# Test BaseModel for Anthropic-style function call
class AnthropicToolCall(BaseModel):
    type: str = "tool_use"
    id: str = "toolu_123456"
    name: str
    input: dict


def test_openai_function_calls():
    """Test OpenAI-style function calls"""
    print("=== Testing OpenAI Function Calls ===")

    tool = BaseTool(tools=[get_current_weather, calculate_sum])

    # OpenAI response format
    openai_response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "get_current_weather",
                                "arguments": '{"location": "Boston", "unit": "fahrenheit"}',
                            },
                        }
                    ]
                }
            }
        ]
    }

    try:
        results = tool.execute_function_calls_from_api_response(
            openai_response
        )
        print("OpenAI Response Results:")
        for result in results:
            print(f"  {result}")
        print()
    except Exception as e:
        print(f"Error with OpenAI response: {e}")
        print()


def test_anthropic_function_calls():
    """Test Anthropic-style function calls"""
    print("=== Testing Anthropic Function Calls ===")

    tool = BaseTool(tools=[get_current_weather, calculate_sum])

    # Anthropic response format
    anthropic_response = {
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_123456",
                "name": "calculate_sum",
                "input": {"a": 15, "b": 25},
            }
        ]
    }

    try:
        results = tool.execute_function_calls_from_api_response(
            anthropic_response
        )
        print("Anthropic Response Results:")
        for result in results:
            print(f"  {result}")
        print()
    except Exception as e:
        print(f"Error with Anthropic response: {e}")
        print()


def test_anthropic_basemodel():
    """Test Anthropic BaseModel function calls"""
    print("=== Testing Anthropic BaseModel Function Calls ===")

    tool = BaseTool(tools=[get_current_weather, calculate_sum])

    # BaseModel object (as would come from Anthropic)
    anthropic_tool_call = AnthropicToolCall(
        name="get_current_weather",
        input={"location": "San Francisco", "unit": "celsius"},
    )

    try:
        results = tool.execute_function_calls_from_api_response(
            anthropic_tool_call
        )
        print("Anthropic BaseModel Results:")
        for result in results:
            print(f"  {result}")
        print()
    except Exception as e:
        print(f"Error with Anthropic BaseModel: {e}")
        print()


def test_list_of_basemodels():
    """Test list of BaseModel function calls"""
    print("=== Testing List of BaseModel Function Calls ===")

    tool = BaseTool(tools=[get_current_weather, calculate_sum])

    # List of BaseModel objects
    tool_calls = [
        AnthropicToolCall(
            name="get_current_weather",
            input={"location": "New York", "unit": "fahrenheit"},
        ),
        AnthropicToolCall(
            name="calculate_sum", input={"a": 10, "b": 20}
        ),
    ]

    try:
        results = tool.execute_function_calls_from_api_response(
            tool_calls
        )
        print("List of BaseModel Results:")
        for result in results:
            print(f"  {result}")
        print()
    except Exception as e:
        print(f"Error with list of BaseModels: {e}")
        print()


def test_format_detection():
    """Test format detection for different response types"""
    print("=== Testing Format Detection ===")

    tool = BaseTool()

    # Test different response formats
    test_cases = [
        {
            "name": "OpenAI Format",
            "response": {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "test",
                                        "arguments": "{}",
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
        },
        {
            "name": "Anthropic Format",
            "response": {
                "content": [
                    {"type": "tool_use", "name": "test", "input": {}}
                ]
            },
        },
        {
            "name": "Anthropic BaseModel",
            "response": AnthropicToolCall(name="test", input={}),
        },
        {
            "name": "Generic Format",
            "response": {"name": "test", "arguments": {}},
        },
    ]

    for test_case in test_cases:
        format_type = tool.detect_api_response_format(
            test_case["response"]
        )
        print(f"  {test_case['name']}: {format_type}")

    print()


if __name__ == "__main__":
    print("Testing Modified Function Call Execution\n")

    test_format_detection()
    test_openai_function_calls()
    test_anthropic_function_calls()
    test_anthropic_basemodel()
    test_list_of_basemodels()

    print("=== All Tests Complete ===")
