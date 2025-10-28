"""
Example demonstrating network error handling in LiteLLM wrapper.

This example shows how the LiteLLM wrapper handles network connectivity issues
and provides helpful error messages to guide users to use local models like Ollama
when internet connection is unavailable.
"""

from swarms.utils import LiteLLM, NetworkConnectionError


def example_with_network_handling():
    """
    Example of using LiteLLM with proper network error handling.

    This function demonstrates how to catch NetworkConnectionError
    and handle it appropriately.
    """
    # Initialize LiteLLM with a cloud model
    model = LiteLLM(
        model_name="gpt-4o-mini",
        temperature=0.7,
        max_tokens=1000,
    )

    try:
        # Try to run the model
        response = model.run(
            task="Explain the concept of quantum entanglement in simple terms."
        )
        print(f"Response: {response}")

    except NetworkConnectionError as e:
        # Handle network connectivity issues
        print(f"Network error detected: {e}")
        print("\nFalling back to local model...")

        # Fallback to a local Ollama model
        local_model = LiteLLM(
            model_name="ollama/llama2",
            temperature=0.7,
            max_tokens=1000,
        )

        try:
            response = local_model.run(
                task="Explain the concept of quantum entanglement in simple terms."
            )
            print(f"Local model response: {response}")
        except Exception as local_error:
            print(f"Local model error: {local_error}")
            print("\nMake sure Ollama is installed and running:")
            print("1. Install: https://ollama.ai")
            print("2. Run: ollama pull llama2")
            print("3. Start the server if not running")


def example_check_internet_connection():
    """
    Example of manually checking internet connectivity.

    This function demonstrates how to use the static method
    to check internet connection before attempting API calls.
    """
    # Check if internet is available
    has_internet = LiteLLM.check_internet_connection()

    if has_internet:
        print("✓ Internet connection available")
        model = LiteLLM(model_name="gpt-4o-mini")
    else:
        print("✗ No internet connection detected")
        print("Using local Ollama model instead...")
        model = LiteLLM(model_name="ollama/llama2")

    # Use the model
    try:
        response = model.run(task="What is the meaning of life?")
        print(f"Response: {response}")
    except NetworkConnectionError as e:
        print(f"Error: {e}")


def example_is_local_model():
    """
    Example of checking if a model is a local model.

    This function demonstrates how to determine if a model
    is local or requires internet connectivity.
    """
    # Check various model names
    models = [
        "gpt-4o-mini",
        "ollama/llama2",
        "anthropic/claude-3",
        "ollama/mistral",
        "local/custom-model",
    ]

    for model_name in models:
        is_local = LiteLLM.is_local_model(model_name)
        status = "Local" if is_local else "Cloud"
        print(f"{model_name}: {status}")


def example_with_custom_base_url():
    """
    Example of using LiteLLM with a custom base URL.

    This demonstrates using a local model server with custom base URL.
    """
    # Using Ollama with custom base URL
    model = LiteLLM(
        model_name="ollama/llama2",
        base_url="http://localhost:11434",
        temperature=0.7,
    )

    try:
        response = model.run(task="Write a haiku about programming.")
        print(f"Response: {response}")
    except NetworkConnectionError as e:
        print(f"Connection error: {e}")
        print("\nTroubleshooting:")
        print("- Ensure Ollama is running on localhost:11434")
        print("- Check if the model is loaded: ollama list")
        print("- Try: ollama serve")


if __name__ == "__main__":
    print("=" * 70)
    print("Example 1: Network Error Handling with Fallback")
    print("=" * 70)
    example_with_network_handling()

    print("\n" + "=" * 70)
    print("Example 2: Manual Internet Connection Check")
    print("=" * 70)
    example_check_internet_connection()

    print("\n" + "=" * 70)
    print("Example 3: Check if Model is Local")
    print("=" * 70)
    example_is_local_model()

    print("\n" + "=" * 70)
    print("Example 4: Custom Base URL")
    print("=" * 70)
    example_with_custom_base_url()
