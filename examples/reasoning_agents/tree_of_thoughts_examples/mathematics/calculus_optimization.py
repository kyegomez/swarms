"""Calculus: the 1 litre can with the smallest surface area.

Breadth-first search that samples each candidate step in a separate call,
which gives more varied approaches (substitution, Lagrange multipliers, ...)
than asking for all candidates at once. Expected answer: r ~ 5.42 cm,
h ~ 10.84 cm (h = 2r), minimum area ~ 553.6 cm^2.
"""

from swarms import TreeOfThoughts

agent = TreeOfThoughts(
    name="Calculus-Optimizer",
    model_name="gpt-5.4",
    search_algorithm="bfs",
    generation_strategy="sample",
    evaluation_strategy="value",
    num_thoughts=3,
    breadth=2,
    max_depth=4,
    thought_description=(
        "One step of the calculus: write a formula, eliminate a variable "
        "using the constraint, differentiate, solve for a critical point, "
        "confirm it is a minimum, or compute the final numbers."
    ),
    evaluation_criteria=(
        "Check the algebra and the derivative. Is the volume constraint "
        "used correctly? Is the critical point confirmed to be a minimum? "
        "Are the units consistent?"
    ),
)

answer = agent.run(
    "A closed cylindrical can must hold exactly 1 litre (1000 cm^3). "
    "Find the radius and height that minimise its total surface area, "
    "and give that minimum area."
)
print(f"Answer: {answer}\n")

result = agent.last_result
for number, step in enumerate(result.steps, 1):
    print(f"{number}. {step}")
print(
    f"\nsolved={result.solved} nodes_expanded={result.nodes_expanded} "
    f"llm_calls={result.llm_calls} tokens={result.usage['total_tokens']}"
)
