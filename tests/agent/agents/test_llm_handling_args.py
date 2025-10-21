from swarms.structs.agent import Agent


def test_llm_handling_args_kwargs():
    """Test that llm_handling properly handles both args and kwargs."""

    # Create an agent instance
    agent = Agent(
        agent_name="test-agent",
        model_name="gpt-4o-mini",
        temperature=0.7,
        max_tokens=1000,
    )

    # Test 1: Call llm_handling with kwargs
    print("Test 1: Testing kwargs handling...")
    try:
        # This should work and add the kwargs to additional_args
        agent.llm_handling(top_p=0.9, frequency_penalty=0.1)
        print("✓ kwargs handling works")
    except Exception as e:
        print(f"✗ kwargs handling failed: {e}")

    # Test 2: Call llm_handling with args (dictionary)
    print("\nTest 2: Testing args handling with dictionary...")
    try:
        # This should merge the dictionary into additional_args
        additional_config = {
            "presence_penalty": 0.2,
            "logit_bias": {"123": 1},
        }
        agent.llm_handling(additional_config)
        print("✓ args handling with dictionary works")
    except Exception as e:
        print(f"✗ args handling with dictionary failed: {e}")

    # Test 3: Call llm_handling with both args and kwargs
    print("\nTest 3: Testing both args and kwargs...")
    try:
        # This should handle both
        additional_config = {"presence_penalty": 0.3}
        agent.llm_handling(
            additional_config, top_p=0.8, frequency_penalty=0.2
        )
        print("✓ combined args and kwargs handling works")
    except Exception as e:
        print(f"✗ combined args and kwargs handling failed: {e}")

    # Test 4: Call llm_handling with non-dictionary args
    print("\nTest 4: Testing non-dictionary args...")
    try:
        # This should store args under 'additional_args' key
        agent.llm_handling(
            "some_string", 123, ["list", "of", "items"]
        )
        print("✓ non-dictionary args handling works")
    except Exception as e:
        print(f"✗ non-dictionary args handling failed: {e}")


if __name__ == "__main__":
    test_llm_handling_args_kwargs()
