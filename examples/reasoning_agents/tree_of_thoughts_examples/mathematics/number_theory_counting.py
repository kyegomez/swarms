"""Number theory: count the n <= 1000 for which 7 divides n^2 + n + 1.

Breadth-first search that keeps three branches per level and averages two
evaluator ratings per candidate, so one over-optimistic rating cannot steer
the beam. Expected answer: 286.
"""

from swarms import TreeOfThoughts

agent = TreeOfThoughts(
    name="Number-Theory-Solver",
    model_name="gpt-5.4",
    search_algorithm="bfs",
    generation_strategy="propose",
    evaluation_strategy="value",
    num_thoughts=3,
    breadth=3,
    max_depth=4,
    n_evaluate_samples=2,
    thought_description=(
        "One mathematical deduction with its computation shown, e.g. "
        "reducing the condition modulo 7, testing residues, or counting "
        "the integers in a residue class up to 1000."
    ),
    evaluation_criteria=(
        "Check the modular arithmetic and every count exactly. A step "
        "that tests residues must cover all seven. Penalize off-by-one "
        "errors at the boundaries 1 and 1000."
    ),
)

answer = agent.run(
    "How many positive integers n with n <= 1000 make n^2 + n + 1 "
    "divisible by 7?"
)
print(f"Answer: {answer}\n")

result = agent.last_result
for number, step in enumerate(result.steps, 1):
    print(f"{number}. {step}")
print(
    f"\nsolved={result.solved} nodes_expanded={result.nodes_expanded} "
    f"llm_calls={result.llm_calls} tokens={result.usage['total_tokens']}"
)
