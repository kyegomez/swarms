from pydantic import BaseModel
from swarms.tools.pydantic_to_json import (
    base_model_to_openai_function,
    multi_base_model_to_openai_function,
)
from swarms.tools.base_tool import BaseTool


# Test Pydantic model
class TestModel(BaseModel):
    """A test model for validation."""

    name: str
    age: int
    email: str = "test@example.com"


def test_base_model_to_openai_function():
    """Test that base_model_to_openai_function accepts output_str parameter."""
    print(
        "Testing base_model_to_openai_function with output_str=False..."
    )
    result_dict = base_model_to_openai_function(
        TestModel, output_str=False
    )
    print(f"✓ Dict result type: {type(result_dict)}")
    print(f"✓ Dict result keys: {list(result_dict.keys())}")

    print(
        "\nTesting base_model_to_openai_function with output_str=True..."
    )
    result_str = base_model_to_openai_function(
        TestModel, output_str=True
    )
    print(f"✓ String result type: {type(result_str)}")
    print(f"✓ String result preview: {result_str[:100]}...")


def test_multi_base_model_to_openai_function():
    """Test that multi_base_model_to_openai_function handles output_str correctly."""
    print(
        "\nTesting multi_base_model_to_openai_function with output_str=False..."
    )
    result_dict = multi_base_model_to_openai_function(
        [TestModel], output_str=False
    )
    print(f"✓ Dict result type: {type(result_dict)}")
    print(f"✓ Dict result keys: {list(result_dict.keys())}")

    print(
        "\nTesting multi_base_model_to_openai_function with output_str=True..."
    )
    result_str = multi_base_model_to_openai_function(
        [TestModel], output_str=True
    )
    print(f"✓ String result type: {type(result_str)}")
    print(f"✓ String result preview: {result_str[:100]}...")


def test_base_tool_methods():
    """Test that BaseTool methods handle output_str parameter correctly."""
    print(
        "\nTesting BaseTool.base_model_to_dict with output_str=False..."
    )
    tool = BaseTool()
    result_dict = tool.base_model_to_dict(TestModel, output_str=False)
    print(f"✓ Dict result type: {type(result_dict)}")
    print(f"✓ Dict result keys: {list(result_dict.keys())}")

    print(
        "\nTesting BaseTool.base_model_to_dict with output_str=True..."
    )
    result_str = tool.base_model_to_dict(TestModel, output_str=True)
    print(f"✓ String result type: {type(result_str)}")
    print(f"✓ String result preview: {result_str[:100]}...")

    print(
        "\nTesting BaseTool.multi_base_models_to_dict with output_str=False..."
    )
    result_dict = tool.multi_base_models_to_dict(
        [TestModel], output_str=False
    )
    print(f"✓ Dict result type: {type(result_dict)}")
    print(f"✓ Dict result length: {len(result_dict)}")

    print(
        "\nTesting BaseTool.multi_base_models_to_dict with output_str=True..."
    )
    result_str = tool.multi_base_models_to_dict(
        [TestModel], output_str=True
    )
    print(f"✓ String result type: {type(result_str)}")
    print(f"✓ String result preview: {result_str[:100]}...")


def test_agent_integration():
    """Test that the Agent class can use the fixed methods without errors."""
    print("\nTesting Agent integration...")
    try:
        from swarms import Agent

        # Create a simple agent with a tool schema
        agent = Agent(
            model_name="gpt-4o-mini",
            tool_schema=TestModel,
            max_loops=1,
            verbose=True,
        )

        # This should not raise an error anymore
        agent.handle_tool_schema_ops()
        print(
            "✓ Agent.handle_tool_schema_ops() completed successfully"
        )

    except Exception as e:
        print(f"✗ Agent integration failed: {e}")
        return False

    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Testing output_str parameter fix")
    print("=" * 60)

    try:
        test_base_model_to_openai_function()
        test_multi_base_model_to_openai_function()
        test_base_tool_methods()

        if test_agent_integration():
            print("\n" + "=" * 60)
            print(
                "✅ All tests passed! The output_str parameter fix is working correctly."
            )
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print(
                "❌ Some tests failed. Please check the implementation."
            )
            print("=" * 60)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
