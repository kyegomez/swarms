"""Proof writing: prove that sqrt(2) + sqrt(3) is irrational.

Depth-first search with a strict threshold: any step the evaluator doubts
(score below 0.7) is pruned and the search backtracks to try another line of
argument. A custom system prompt gives every call a mathematician's persona.
"""

from swarms import TreeOfThoughts

agent = TreeOfThoughts(
    name="Proof-Writer",
    model_name="gpt-5.4",
    system_prompt=(
        "You are a rigorous research mathematician. Every claim you make "
        "follows from earlier steps or from a standard, named theorem, "
        "and you never assume what you are asked to prove."
    ),
    search_algorithm="dfs",
    generation_strategy="propose",
    evaluation_strategy="value",
    num_thoughts=2,
    max_depth=6,
    value_threshold=0.7,
    thought_description=(
        "One step of the proof: a single precise deduction together with "
        "its justification. The last step concludes the proof."
    ),
    evaluation_criteria=(
        "Does the step follow rigorously from the earlier steps or a "
        "standard theorem? Reject circular reasoning, unjustified claims, "
        "and gaps. A final step must complete a valid proof."
    ),
)

proof = agent.run("Prove that sqrt(2) + sqrt(3) is irrational.")
print(f"Proof:\n{proof}\n")

result = agent.last_result
print(
    f"solved={result.solved} nodes_expanded={result.nodes_expanded} "
    f"llm_calls={result.llm_calls} tokens={result.usage['total_tokens']}"
)
