"""
Max Loops Parameter Refactoring Example

This example demonstrates the max_loops parameter (renamed from max_iterations)
and shows different usage patterns. This change improves code clarity by better
reflecting the actual behavior of the parameter.

Key changes:
- max_iterations → max_loops (better reflects loop-based execution)
- Full backwards compatibility maintained
- Supports both integer values and "auto" for autonomous loops
- Clearer semantic meaning for developers
"""

from swarms import Agent

# Example 1: Traditional fixed number of loops (formerly max_iterations=3)
agent_fixed_loops = Agent(
    agent_name="FixedLoopsAgent",
    system_prompt="You are a helpful assistant that provides step-by-step solutions.",
    model_name="gpt-4o-mini",
    max_loops=3,
)

response1 = agent_fixed_loops.run(
    "Explain how to bake chocolate chip cookies in 5 clear steps."
)
print(response1)

agent_autonomous = Agent(
    agent_name="AutonomousAgent",
    system_prompt="""You are an autonomous problem solver that can break down complex tasks
    and execute them step by step.""",
    model_name="gpt-4o-mini",
    max_loops="auto",
    interactive=False,
)

response2 = agent_autonomous.run(
    "Research the best electric vehicles available in 2026."
)
print(response2)
