"""
Runner script to execute all uvloop examples.

This script demonstrates how to run multiple uvloop-based agent execution examples.
"""

import os
from same_task_example import run_same_task_example
from different_tasks_example import run_different_tasks_example


def run_all_uvloop_examples():
    """
    Execute all uvloop examples.

    Returns:
        Dictionary containing results from all examples
    """
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY environment variable must be set"
        )

    results = {}

    # Run same task example
    results["same_task"] = run_same_task_example()

    # Run different tasks example
    results["different_tasks"] = run_different_tasks_example()

    return results


if __name__ == "__main__":
    all_results = run_all_uvloop_examples()
    # Process results as needed
