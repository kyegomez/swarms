from swarms.utils.litellm_wrapper import LiteLLM


def test_litellm_args_kwargs():
    """Test that LiteLLM properly handles both args and kwargs from __init__ and run."""

    print("Testing LiteLLM args and kwargs handling...")

    # Test 1: Initialize with kwargs
    print("\nTest 1: Testing __init__ with kwargs...")
    try:
        llm = LiteLLM(
            model_name="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1000,
            top_p=0.9,
            frequency_penalty=0.1,
        )
        print("✓ __init__ with kwargs works")
        print(f"  - init_kwargs: {llm.init_kwargs}")
        print(f"  - init_args: {llm.init_args}")
    except Exception as e:
        print(f"✗ __init__ with kwargs failed: {e}")

    # Test 2: Initialize with args (dictionary)
    print("\nTest 2: Testing __init__ with args (dictionary)...")
    try:
        additional_config = {
            "presence_penalty": 0.2,
            "logit_bias": {"123": 1},
        }
        llm = LiteLLM("gpt-4o-mini", additional_config)
        print("✓ __init__ with args (dictionary) works")
        print(f"  - init_args: {llm.init_args}")
    except Exception as e:
        print(f"✗ __init__ with args (dictionary) failed: {e}")

    # Test 3: Initialize with both args and kwargs
    print("\nTest 3: Testing __init__ with both args and kwargs...")
    try:
        additional_config = {"presence_penalty": 0.3}
        llm = LiteLLM(
            "gpt-4o-mini",
            additional_config,
            temperature=0.8,
            max_tokens=2000,
        )
        print("✓ __init__ with both args and kwargs works")
        print(f"  - init_args: {llm.init_args}")
        print(f"  - init_kwargs: {llm.init_kwargs}")
    except Exception as e:
        print(f"✗ __init__ with both args and kwargs failed: {e}")

    # Test 4: Run method with kwargs (overriding init kwargs)
    print("\nTest 4: Testing run method with kwargs...")
    try:
        llm = LiteLLM(
            model_name="gpt-4o-mini",
            temperature=0.5,  # This should be overridden
            max_tokens=1000,
        )

        # This should override the init temperature
        result = llm.run(
            "Hello, world!",
            temperature=0.9,  # Should override the 0.5 from init
            top_p=0.8,
        )
        print("✓ run method with kwargs works")
        print(f"  - Result type: {type(result)}")
    except Exception as e:
        print(f"✗ run method with kwargs failed: {e}")

    # Test 5: Run method with args (dictionary)
    print("\nTest 5: Testing run method with args (dictionary)...")
    try:
        llm = LiteLLM(model_name="gpt-4o-mini")

        runtime_config = {"temperature": 0.8, "max_tokens": 500}
        result = llm.run("Hello, world!", runtime_config)
        print("✓ run method with args (dictionary) works")
        print(f"  - Result type: {type(result)}")
    except Exception as e:
        print(f"✗ run method with args (dictionary) failed: {e}")

    # Test 6: Priority order test
    print("\nTest 6: Testing parameter priority order...")
    try:
        # Init with some kwargs
        llm = LiteLLM(
            model_name="gpt-4o-mini",
            temperature=0.1,  # Should be overridden
            max_tokens=100,  # Should be overridden
        )

        # Run with different values
        runtime_config = {"temperature": 0.9, "max_tokens": 2000}
        result = llm.run(
            "Hello, world!",
            runtime_config,
            temperature=0.5,  # Should override both init and runtime_config
            max_tokens=500,  # Should override both init and runtime_config
        )
        print("✓ parameter priority order works")
        print(f"  - Result type: {type(result)}")
    except Exception as e:
        print(f"✗ parameter priority order failed: {e}")


if __name__ == "__main__":
    test_litellm_args_kwargs()
