"""Kinematics: where a ball thrown from a cliff lands.

Breadth-first search whose evaluator checks signs, units and the choice of
kinematic equation at every step, so a sign slip in the quadratic for time is
pruned before it propagates. Expected answer: about 38.0 m (flight time
about 2.93 s).
"""

from swarms import TreeOfThoughts

agent = TreeOfThoughts(
    name="Kinematics-Solver",
    model_name="gpt-5.4",
    search_algorithm="bfs",
    generation_strategy="propose",
    evaluation_strategy="value",
    num_thoughts=3,
    breadth=2,
    max_depth=4,
    thought_description=(
        "One physics step with units: resolve a velocity into components, "
        "write one kinematic equation with its known values, solve it for "
        "one unknown, or combine earlier results."
    ),
    evaluation_criteria=(
        "Check signs, units, the choice of kinematic equation and the "
        "arithmetic. The negative root of a quadratic in time is not "
        "physical. Does the magnitude make physical sense?"
    ),
)

answer = agent.run(
    "A ball is thrown from the edge of a 20 m high cliff at 15 m/s, 30 "
    "degrees above the horizontal. Taking g = 9.8 m/s^2 and ignoring air "
    "resistance, how far from the base of the cliff does it land?"
)
print(f"Answer: {answer}\n")

result = agent.last_result
for number, step in enumerate(result.steps, 1):
    print(f"{number}. {step}")
print(
    f"\nsolved={result.solved} nodes_expanded={result.nodes_expanded} "
    f"llm_calls={result.llm_calls} tokens={result.usage['total_tokens']}"
)
