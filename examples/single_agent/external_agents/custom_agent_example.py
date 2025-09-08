import os

from dotenv import load_dotenv

from swarms.structs.custom_agent import CustomAgent

load_dotenv()

# Example usage with Anthropic API
if __name__ == "__main__":
    # Initialize the agent for Anthropic API
    anthropic_agent = CustomAgent(
        base_url="https://api.anthropic.com",
        endpoint="v1/messages",
        headers={
            "x-api-key": os.getenv("ANTHROPIC_API_KEY"),
            "anthropic-version": "2023-06-01",
        },
    )

    # Example payload for Anthropic API
    payload = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1000,
        "messages": [
            {
                "role": "user",
                "content": "Hello! Can you explain what artaddificial intelligence is?",
            }
        ],
    }

    # Make the request
    try:
        response = anthropic_agent.run(payload)
        print("Anthropic API Response:")
        print(response)
        print(type(response))
    except Exception as e:
        print(f"Error: {e}")
