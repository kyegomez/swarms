#!/usr/bin/env python3
"""
Example usage of the modified execute_function_calls_from_api_response method
with the exact response structure from tool_schema.py
"""

from swarms.tools.base_tool import BaseTool


def get_current_weather(location: str, unit: str = "celsius") -> dict:
    """Get the current weather in a given location"""
    return {
        "location": location,
        "temperature": "22" if unit == "celsius" else "72",
        "unit": unit,
        "condition": "sunny",
        "description": f"The weather in {location} is sunny with a temperature of {'22°C' if unit == 'celsius' else '72°F'}",
    }


def main():
    """
    Example of using the modified BaseTool with a LiteLLM response
    that contains Anthropic function calls as BaseModel objects
    """

    # Set up the BaseTool with your functions
    tool = BaseTool(tools=[get_current_weather], verbose=True)

    # Simulate the response you get from LiteLLM (from your tool_schema.py output)
    # In real usage, this would be: response = completion(...)

    # For this example, let's simulate the exact response structure
    # The response.choices[0].message.tool_calls contains BaseModel objects
    print("=== Simulating LiteLLM Response Processing ===")

    # Option 1: Process the entire response object
    # (This would be the actual ModelResponse object from LiteLLM)
    mock_response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        # This would actually be a ChatCompletionMessageToolCall BaseModel object
                        # but we'll simulate the structure here
                        {
                            "index": 1,
                            "function": {
                                "arguments": '{"location": "Boston", "unit": "fahrenheit"}',
                                "name": "get_current_weather",
                            },
                            "id": "toolu_019vcXLipoYHzd1e1HUYSSaa",
                            "type": "function",
                        }
                    ]
                }
            }
        ]
    }

    print("Processing mock response:")
    try:
        results = tool.execute_function_calls_from_api_response(
            mock_response
        )
        print("Results:")
        for i, result in enumerate(results):
            print(f"  Function call {i+1}:")
            print(f"    {result}")
    except Exception as e:
        print(f"Error processing response: {e}")

    print("\n" + "=" * 50)

    # Option 2: Process just the tool_calls list
    # (If you extract tool_calls from response.choices[0].message.tool_calls)
    print("Processing just tool_calls:")

    tool_calls = mock_response["choices"][0]["message"]["tool_calls"]

    try:
        results = tool.execute_function_calls_from_api_response(
            tool_calls
        )
        print("Results from tool_calls:")
        for i, result in enumerate(results):
            print(f"  Function call {i+1}:")
            print(f"    {result}")
    except Exception as e:
        print(f"Error processing tool_calls: {e}")

    print("\n" + "=" * 50)

    # Option 3: Show format detection
    print("Format detection:")
    format_type = tool.detect_api_response_format(mock_response)
    print(f"  Full response format: {format_type}")

    format_type_tools = tool.detect_api_response_format(tool_calls)
    print(f"  Tool calls format: {format_type_tools}")


if __name__ == "__main__":
    main()
