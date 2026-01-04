"""
Council as a Judge Example

This example demonstrates CouncilAsAJudge for multi-dimensional evaluation
of task responses across multiple specialized judge agents.
"""

from swarms import CouncilAsAJudge

council = CouncilAsAJudge(
    name="Evaluation-Council",
    description="Evaluates responses across multiple dimensions",
    model_name="gpt-4o-mini",
    max_loops=1,
)

task_response = "Artificial intelligence will transform healthcare by enabling early disease detection, personalized treatment plans, and reducing medical errors through advanced pattern recognition."

result = council.run(task_response)

print("Council as a Judge Result:")
print(result)
