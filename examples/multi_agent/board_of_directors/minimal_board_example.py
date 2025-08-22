"""
Minimal Board of Directors Example

This example demonstrates the most basic Board of Directors swarm setup
with minimal configuration and agents.

To run this example:
1. Make sure you're in the root directory of the swarms project
2. Run: python examples/multi_agent/board_of_directors/minimal_board_example.py
"""

from swarms.structs.board_of_directors_swarm import (
    BoardOfDirectorsSwarm,
)
from swarms.structs.agent import Agent


def run_minimal_example() -> str:
    """Run a minimal Board of Directors example."""
    # Create a single agent
    agent = Agent(
        agent_name="General_Agent",
        agent_description="General purpose agent",
        model_name="gpt-4o-mini",
        max_loops=1,
    )

    # Create minimal swarm
    board_swarm = BoardOfDirectorsSwarm(
        name="Minimal_Board",
        agents=[agent],
        verbose=False,
    )

    # Execute minimal task
    task = "Provide a brief overview of artificial intelligence."
    return board_swarm.run(task=task)


def main() -> None:
    """Main function to run the minimal example."""

    try:
        result = run_minimal_example()
        return result
    except Exception:
        pass


if __name__ == "__main__":
    main()
