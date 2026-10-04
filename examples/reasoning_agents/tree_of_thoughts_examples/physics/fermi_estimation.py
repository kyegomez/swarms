"""Fermi estimation: the kinetic energy of all cars on US roads at rush hour.

There is no single correct path here, only more and less defensible
assumptions, so candidates are sampled independently and compared by vote
rather than rated on an absolute scale. Three votes per comparison keep more
than one branch alive. Reasonable answers land near 10^13 J.
"""

from swarms import TreeOfThoughts

agent = TreeOfThoughts(
    name="Fermi-Estimator",
    model_name="gpt-5.4",
    search_algorithm="bfs",
    generation_strategy="sample",
    evaluation_strategy="vote",
    num_thoughts=3,
    breadth=2,
    max_depth=4,
    n_evaluate_samples=3,
    thought_description=(
        "One estimation step: a single quantity with a stated, justified "
        "assumption (for example the number of cars moving, a typical "
        "mass, a typical speed), or a combination of earlier quantities."
    ),
    evaluation_criteria=(
        "Prefer realistic, explicitly justified assumptions, correct "
        "arithmetic and units, and steady progress toward one number in "
        "joules."
    ),
)

answer = agent.run(
    "Estimate, to an order of magnitude, the total kinetic energy in "
    "joules of all cars moving on United States roads during a weekday "
    "rush hour. State your assumptions."
)
print(f"Answer: {answer}\n")

result = agent.last_result
for number, step in enumerate(result.steps, 1):
    print(f"{number}. {step}")
print(
    f"\nsolved={result.solved} nodes_expanded={result.nodes_expanded} "
    f"llm_calls={result.llm_calls} tokens={result.usage['total_tokens']}"
)
