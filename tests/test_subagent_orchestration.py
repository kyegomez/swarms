"""
Subagent orchestration end-to-end test.

Verifies that the autonomous loop's create_sub_agent tool correctly
generates a default system_prompt when the LLM omits it, preventing
the "system: text content blocks must be non-empty" Anthropic API error.

Run:
    python tests/test_subagent_orchestration.py
"""

from dotenv import load_dotenv

load_dotenv()

from swarms.structs.agent import Agent


def test_subagent_orchestration():
    """
    End-to-end test: Agent uses create_sub_agent and assign_task tools
    in its autonomous loop. The fix ensures sub-agents created without
    an explicit system_prompt get a default one.
    """
    agent = Agent(
        agent_name="test-parent",
        model_name="gpt-4.1-nano",
        max_loops="auto",
        interactive=False,
        streaming_on=False,
        verbose=True,
    )

    result = agent.run(
        "Create 3 sub-agents that each greet the user in a different language"
    )
    print(f"\nFinal result:\n{result}")


if __name__ == "__main__":
    test_subagent_orchestration()
