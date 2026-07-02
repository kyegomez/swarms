"""
FuguAgent: A multi-agent system that behaves as a single model.

.. deprecated::
    This example imports from the new location:
    ``examples.multi_agent.fugu_agent``

    Please update your imports to use:
        from examples.multi_agent.fugu_agent import FuguAgent

FuguAgent uses a dedicated coordinator model that decides which worker to use,
what role to assign, and what instruction to give — dynamically, per step.

The coordinator uses tool calling (not text parsing) to commit to each step's
AgentTask {role, worker, instruction, visibility}. Workers are ranked by
capability and the most powerful models are assigned to the hardest tasks.

Usage:
    from examples.multi_agent.fugu_agent import FuguAgent

    # Auto-detects API keys and builds worker pool from available models
    agent = FuguAgent(max_turns=5, verbose=True)

    # Or specify workers explicitly
    from swarms import Agent
    workers = [
        Agent(agent_name="coder", model_name="gpt-4o"),
        Agent(agent_name="researcher", model_name="claude-sonnet-4-5"),
    ]
    agent = FuguAgent(workers=workers, max_turns=5)

    result = agent.run("Write and test a function that checks if a string is a palindrome.")
"""

import warnings

warnings.warn(
    "examples.single_agent.fugu_example is deprecated. "
    "Import from examples.multi_agent.fugu_agent instead.",
    DeprecationWarning,
    stacklevel=2,
)

from examples.multi_agent.fugu_agent import FuguAgent


def main():
    agent = FuguAgent(
        coordinator_model="gpt-4o-mini",
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
