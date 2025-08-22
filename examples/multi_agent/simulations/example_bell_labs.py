"""
Bell Labs Research Simulation Example

This example demonstrates how to use the BellLabsSwarm to simulate
collaborative research among famous physicists.
"""

from swarms.sims.bell_labs import (
    run_bell_labs_research,
)


def main():
    """
    Run the Bell Labs research simulation.

    This example asks the research question:
    "Why doesn't physics take a vacation? Why are the laws of physics consistent?"
    """

    research_question = """
    Why doesn't physics take a vacation? Why are the laws of physics consistent across time and space?
    Explore the philosophical and scientific foundations for the uniformity and invariance of physical laws.
    Consider both theoretical explanations and any empirical evidence or challenges to this consistency.
    """

    # Run the research simulation
    results = run_bell_labs_research(
        research_question=research_question,
        max_loops=1,
        model_name="claude-3-5-sonnet-20240620",
        verbose=True,
    )

    print(results)


if __name__ == "__main__":
    main()
