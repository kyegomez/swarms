"""
FuguAgent with explicit worker agents example.

This example demonstrates how to pass explicit Agent instances as workers,
giving you full control over each worker's configuration including
agent_name, model_name, system_prompt, and tools.

Usage:
    python example_with_workers.py
"""

from dotenv import load_dotenv

from swarms import Agent
from examples.multi_agent.fugu_agent import FuguAgent

load_dotenv()


def main() -> None:
    """Run FuguAgent with explicitly configured workers."""
    workers = [
        Agent(
            agent_name="coder",
            model_name="gpt-5.1",
            system_prompt="You are an expert programmer. Write clean, well-documented code.",
            max_loops=1,
        ),
        Agent(
            agent_name="researcher",
            model_name="claude-sonnet-4-5",
            system_prompt="You are a thorough researcher. Find the most relevant information.",
            max_loops=1,
        ),
        Agent(
            agent_name="writer",
            model_name="gpt-5.1",
            system_prompt="You are a skilled technical writer. Write clear, concise content.",
            max_loops=1,
        ),
    ]

    agent = FuguAgent(
        coordinator_model="gpt-5.4",
        workers=workers,
        max_turns=5,
        verbose=True,
    )

    result = agent.run(
        "Write and test a function that checks if a string is a palindrome."
    )
    print("\n=== Final Answer ===")
    print(result)


if __name__ == "__main__":
    main()
