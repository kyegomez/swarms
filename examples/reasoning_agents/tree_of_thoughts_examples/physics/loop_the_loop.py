"""Energy conservation: the minimum release height for a loop-the-loop.

Depth-first search that commits to the most promising line of physics first
(force balance at the top, then energy conservation) and backtracks if a step
uses the wrong condition, such as requiring zero speed at the top. Expected
answer: 2.0 m, which is 5R/2.
"""

from swarms import TreeOfThoughts

agent = TreeOfThoughts(
    name="Mechanics-Solver",
    model_name="gpt-5.4",
    search_algorithm="dfs",
    generation_strategy="propose",
    evaluation_strategy="value",
    num_thoughts=3,
    max_depth=4,
    thought_description=(
        "One physical principle applied: a force balance at one point of "
        "the loop, an energy-conservation equation between two positions, "
        "or solving an equation for one quantity."
    ),
    evaluation_criteria=(
        "Is the condition for staying on the track right (normal force at "
        "least zero at the top)? Is energy conserved between the right two "
        "points, with heights from the same reference? Check the algebra "
        "and units."
    ),
)

answer = agent.run(
    "A small block slides without friction down a ramp and into a "
    "vertical circular loop of radius 0.8 m. From what minimum height "
    "above the bottom of the loop must it be released so that it stays on "
    "the track at the top of the loop? Take g = 9.8 m/s^2."
)
print(f"Answer: {answer}\n")

result = agent.last_result
for number, step in enumerate(result.steps, 1):
    print(f"{number}. {step}")
print(
    f"\nsolved={result.solved} nodes_expanded={result.nodes_expanded} "
    f"llm_calls={result.llm_calls} tokens={result.usage['total_tokens']}"
)
