"""
Planner-Generator-Evaluator Harness — Custom Agents with Tools

Demonstrates passing pre-configured agents with tools to the PGE harness.
The Generator gets a file-writing tool, and the Evaluator gets a
file-reading tool to verify the Generator's output on disk.

To run:
    python examples/multi_agent/planner_generator_evaluator/pge_with_tools_example.py
"""

import os
import tempfile

from swarms import Agent, PlannerGeneratorEvaluator

WORK_DIR = tempfile.mkdtemp(prefix="pge_tools_")


def write_file(filename: str, content: str) -> str:
    """Write content to a file in the working directory.

    Args:
        filename: Name of the file to create.
        content: Text content to write.

    Returns:
        Confirmation message with the file path.
    """
    path = os.path.join(WORK_DIR, filename)
    with open(path, "w") as f:
        f.write(content)
    return f"File written: {path} ({len(content)} chars)"


def read_file(filename: str) -> str:
    """Read content from a file in the working directory.

    Args:
        filename: Name of the file to read.

    Returns:
        The file contents, or an error message if not found.
    """
    path = os.path.join(WORK_DIR, filename)
    if not os.path.exists(path):
        return f"File not found: {path}"
    with open(path, "r") as f:
        return f.read()


def list_files() -> str:
    """List all files in the working directory.

    Returns:
        Newline-separated list of filenames.
    """
    files = os.listdir(WORK_DIR)
    return "\n".join(files) if files else "No files found"


if __name__ == "__main__":
    generator = Agent(
        agent_name="PGE-Generator",
        agent_description="Generator that can write files to disk",
        model_name="gpt-5.4",
        max_loops=1,
        tools=[write_file, list_files],
    )

    evaluator = Agent(
        agent_name="PGE-Evaluator",
        agent_description="Evaluator that can read and verify files on disk",
        model_name="gpt-5.4",
        max_loops=1,
        tools=[read_file, list_files],
    )

    harness = PlannerGeneratorEvaluator(
        model_name="gpt-5.4",
        generator_agent=generator,
        evaluator_agent=evaluator,
        max_steps=2,
        max_retries_per_step=1,
        output_type="final",
        verbose=True,
    )

    result = harness.run(
        "Create a Python utility module with helper functions for string manipulation"
    )

    print(result)
    print(f"\nFiles created in {WORK_DIR}:")
    for f in os.listdir(WORK_DIR):
        print(f"  - {f}")
