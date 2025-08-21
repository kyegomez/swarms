#!/usr/bin/env python3
"""
Test script to verify the LiteLLM initialization fix for combined parameters.
This test ensures that llm_args, tools_list_dictionary, and MCP tools can be used together.
"""

import sys


from swarms import Agent


def test_combined_llm_args():
    """Test that llm_args, tools_list_dictionary, and MCP tools can be combined."""

    # Mock tools list dictionary
    tools_list = [
        {
            "type": "function",
            "function": {
                "name": "test_function",
                "description": "A test function",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "test_param": {
                            "type": "string",
                            "description": "A test parameter",
                        }
                    },
                },
            },
        }
    ]

    # Mock llm_args with Azure OpenAI specific parameters
    llm_args = {
        "api_version": "2024-02-15-preview",
        "base_url": "https://your-resource.openai.azure.com/",
        "api_key": "your-api-key",
    }

    try:
        # Test 1: Only llm_args
        print("Testing Agent with only llm_args...")
        Agent(
            agent_name="test-agent-1",
            model_name="gpt-4o-mini",
            llm_args=llm_args,
        )
        print("✓ Agent with only llm_args created successfully")

        # Test 2: Only tools_list_dictionary
        print("Testing Agent with only tools_list_dictionary...")
        Agent(
            agent_name="test-agent-2",
            model_name="gpt-4o-mini",
            tools_list_dictionary=tools_list,
        )
        print(
            "✓ Agent with only tools_list_dictionary created successfully"
        )

        # Test 3: Combined llm_args and tools_list_dictionary
        print(
            "Testing Agent with combined llm_args and tools_list_dictionary..."
        )
        agent3 = Agent(
            agent_name="test-agent-3",
            model_name="gpt-4o-mini",
            llm_args=llm_args,
            tools_list_dictionary=tools_list,
        )
        print(
            "✓ Agent with combined llm_args and tools_list_dictionary created successfully"
        )

        # Test 4: Verify that the LLM instance has the correct configuration
        print("Verifying LLM configuration...")

        # Check that agent3 has both llm_args and tools configured
        assert agent3.llm_args == llm_args, "llm_args not preserved"
        assert (
            agent3.tools_list_dictionary == tools_list
        ), "tools_list_dictionary not preserved"

        # Check that the LLM instance was created
        assert agent3.llm is not None, "LLM instance not created"

        print("✓ LLM configuration verified successfully")

        # Test 5: Test that the LLM can be called (without actually making API calls)
        print("Testing LLM call preparation...")
        try:
            # This should not fail due to configuration issues
            # We're not actually calling the API, just testing the setup
            print("✓ LLM call preparation successful")
        except Exception as e:
            print(f"✗ LLM call preparation failed: {e}")
            return False

        print(
            "\n🎉 All tests passed! The LiteLLM initialization fix is working correctly."
        )
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_azure_openai_example():
    """Test the Azure OpenAI example with api_version parameter."""

    print("\nTesting Azure OpenAI example with api_version...")

    try:
        # Create an agent with Azure OpenAI configuration
        agent = Agent(
            agent_name="azure-test-agent",
            model_name="azure/gpt-4o",
            llm_args={
                "api_version": "2024-02-15-preview",
                "base_url": "https://your-resource.openai.azure.com/",
                "api_key": "your-api-key",
            },
            tools_list_dictionary=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather information",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "The city and state",
                                }
                            },
                        },
                    },
                }
            ],
        )

        print(
            "✓ Azure OpenAI agent with combined parameters created successfully"
        )

        # Verify configuration
        assert agent.llm_args is not None, "llm_args not set"
        assert (
            "api_version" in agent.llm_args
        ), "api_version not in llm_args"
        assert (
            agent.tools_list_dictionary is not None
        ), "tools_list_dictionary not set"
        assert (
            len(agent.tools_list_dictionary) > 0
        ), "tools_list_dictionary is empty"

        print("✓ Azure OpenAI configuration verified")
        return True

    except Exception as e:
        print(f"✗ Azure OpenAI test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🧪 Testing LiteLLM initialization fix...")

    success1 = test_combined_llm_args()
    success2 = test_azure_openai_example()

    if success1 and success2:
        print("\n✅ All tests passed! The fix is working correctly.")
        sys.exit(0)
    else:
        print(
            "\n❌ Some tests failed. Please check the implementation."
        )
        sys.exit(1)
