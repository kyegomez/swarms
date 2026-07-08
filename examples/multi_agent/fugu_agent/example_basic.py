"""
Basic FuguAgent usage example.

This example demonstrates minimal FuguAgent usage with auto-detected API keys.
The agent will automatically detect OPENAI_API_KEY, ANTHROPIC_API_KEY, or
GOOGLE_API_KEY and build a worker pool from available models.

Usage:
    python example_basic.py
"""

from dotenv import load_dotenv

from examples.multi_agent.fugu_agent import FuguAgent

load_dotenv()


def main() -> None:
    """Run a simple FuguAgent task."""
    agent = FuguAgent(
        coordinator_model="gpt-5",
        max_turns=5,
        verbose=True,
    )

    result = agent.run(
        "How to solve Turing's halting problem?"
    )
    print("\n=== Final Answer ===")
    print(result)


if __name__ == "__main__":
    main()
