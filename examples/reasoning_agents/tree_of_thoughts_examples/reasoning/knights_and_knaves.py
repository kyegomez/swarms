"""Logic puzzle: identify the knights and knaves among three islanders.

Depth-first search suits case analysis: the agent assumes one islander's
type, follows the consequences, and backtracks when the evaluator finds a
contradiction. Expected answer: A is a knight, B is a knave, C is a knight.
"""

from swarms import TreeOfThoughts

agent = TreeOfThoughts(
    name="Logic-Solver",
    model_name="gpt-5.4",
    search_algorithm="dfs",
    generation_strategy="propose",
    evaluation_strategy="value",
    num_thoughts=2,
    max_depth=4,
    thought_description=(
        "One deduction: assume one islander's type and derive what "
        "follows, or rule out a case by exhibiting a contradiction."
    ),
    evaluation_criteria=(
        "Is each deduction logically valid and consistent with every "
        "statement? A case that leads to a contradiction must be ruled "
        "out, not kept. A final answer must be consistent with all three "
        "statements."
    ),
)

answer = agent.run(
    "On an island, knights always tell the truth and knaves always lie. "
    "A says: 'B is a knave.' B says: 'Exactly one of us three is a "
    "knight.' C says: 'B and I are different kinds.' Which of A, B and C "
    "are knights and which are knaves?"
)
print(f"Answer: {answer}\n")

result = agent.last_result
for number, step in enumerate(result.steps, 1):
    print(f"{number}. {step}")
print(
    f"\nsolved={result.solved} nodes_expanded={result.nodes_expanded} "
    f"llm_calls={result.llm_calls} tokens={result.usage['total_tokens']}"
)
