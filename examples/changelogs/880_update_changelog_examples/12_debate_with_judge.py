"""
Debate with Judge Example

This example demonstrates DebateWithJudge architecture where Pro and Con
agents debate a topic with iterative refinement through a judge agent.
"""

from swarms import DebateWithJudge

debate = DebateWithJudge(
    preset_agents=True,
    max_loops=3,
    verbose=True,
)

task = "Should AI be regulated?"
result = debate.run(task)

print("Debate with Judge Result:")
print(result)
