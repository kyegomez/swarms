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
    result_list = multi_base_model_to_openai_function(
        [TestModel], output_str=False
    )
    print(f"✓ List result type: {type(result_list)}")
    print(f"✓ List result length: {len(result_list)}")
    print(f"✓ First schema keys: {list(result_list[0].keys())}")

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
            model_name="gpt-5.4",
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


def test_function_name_is_the_model_name():
    """The emitted schema must name the model, not the metaclass or a field.

    Two bugs compounded here. `type(pydantic_type).__name__` read the
    metaclass — pydantic_type is the class, not an instance — so the name
    was "ModelMetaclass". And the docstring loop's walrus bound `name`,
    so whenever a docstring param matched a property the name became
    whichever param matched last. Either way the LLM was told the tool had
    a name it does not have.
    """
    from pydantic import BaseModel, Field

    from swarms.tools.pydantic_to_json import (
        base_model_to_openai_function,
    )

    class WeatherQuery(BaseModel):
        """Look up the weather.

        Args:
            city: The city to look up.
            units: Temperature units.
        """

        city: str = Field(...)
        units: str = Field("celsius")

    result = base_model_to_openai_function(WeatherQuery)

    assert result["type"] == "function"
    assert result["function"]["name"] == "WeatherQuery"

    # The rename must not cost the per-parameter descriptions the loop
    # exists to attach.
    props = result["function"]["parameters"]["properties"]
    assert props["city"]["description"] == "The city to look up."
    assert props["units"]["description"] == "Temperature units."


def test_no_legacy_envelope_round_trip():
    """Emit modern tools shape directly; BaseTool must not unwrap/rewrap.

    #1848: base_model_to_openai_function used to build
    {function_call, functions:[…]} only for base_model_to_dict to peel
    functions[0] and re-wrap as {type, function}. Both paths must now
    share one modern schema with no envelope keys.
    """
    from swarms.tools.base_tool import BaseTool

    helper = base_model_to_openai_function(TestModel)
    via_tool = BaseTool().base_model_to_dict(TestModel)

    assert helper == via_tool
    assert "function_call" not in helper
    assert "functions" not in helper
    assert helper["type"] == "function"
    assert helper["function"]["name"] == "TestModel"

    multi = multi_base_model_to_openai_function([TestModel])
    assert isinstance(multi, list)
    assert multi[0] == helper
    assert "function_call" not in multi[0]
    assert "functions" not in multi[0]
