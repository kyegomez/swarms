#!/usr/bin/env python3
"""
Simple Example: Function Schema Validation for Different AI Providers
Demonstrates the validation logic for OpenAI, Anthropic, and generic function calling schemas
"""

from swarms.tools.base_tool import BaseTool


def main():
    """Run schema validation examples"""
    print("🔍 Function Schema Validation Examples")
    print("=" * 50)

    # Initialize BaseTool
    tool = BaseTool(verbose=True)

    # Example schemas for different providers

    # 1. OpenAI Function Calling Schema
    print("\n📘 OpenAI Schema Validation")
    print("-" * 30)

    openai_schema = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["location"],
            },
        },
    }

    is_valid = tool.validate_function_schema(openai_schema, "openai")
    print(f"✅ OpenAI schema valid: {is_valid}")

    # 2. Anthropic Tool Schema
    print("\n📗 Anthropic Schema Validation")
    print("-" * 30)

    anthropic_schema = {
        "name": "calculate_sum",
        "description": "Calculate the sum of two numbers",
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {
                    "type": "number",
                    "description": "First number",
                },
                "b": {
                    "type": "number",
                    "description": "Second number",
                },
            },
            "required": ["a", "b"],
        },
    }

    is_valid = tool.validate_function_schema(
        anthropic_schema, "anthropic"
    )
    print(f"✅ Anthropic schema valid: {is_valid}")


if __name__ == "__main__":
    main()
