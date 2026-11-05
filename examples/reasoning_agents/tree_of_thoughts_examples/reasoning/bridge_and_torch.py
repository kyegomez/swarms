"""Planning: get four people across a bridge by torchlight in minimum time.

A classic trap for single-pass reasoning: the obvious plan, where the fastest
person escorts everyone, takes 19 minutes. Depth-first search explores one
crossing per step and backtracks from schedules that can no longer be
optimal. max_expansions caps the cost of a deep search. Expected answer:
17 minutes.
"""

from swarms import TreeOfThoughts

agent = TreeOfThoughts(
    name="Bridge-Planner",
    model_name="gpt-5.4",
    search_algorithm="dfs",
    generation_strategy="propose",
    evaluation_strategy="value",
    num_thoughts=3,
    max_depth=5,
    max_expansions=15,
    thought_description=(
        "One crossing: who crosses, in which direction, how long it takes, "
        "the running total, and who is on each side afterwards."
    ),
    evaluation_criteria=(
        "Is the crossing legal (the torch is on the crossers' side, at "
        "most two people)? Is the running total right? Judge whether the "
        "schedule can still reach the minimum possible total, not merely "
        "some valid total."
    ),
)

answer = agent.run(
    "Four people must cross a narrow bridge at night. They take 1, 2, 5 "
    "and 10 minutes to cross. At most two can cross at once, a pair moves "
    "at the slower person's pace, and their single torch must accompany "
    "every crossing. What is the minimum total time to get everyone "
    "across, and what is the schedule?"
)
print(f"Answer: {answer}\n")

result = agent.last_result
for number, step in enumerate(result.steps, 1):
    print(f"{number}. {step}")
print(
    f"\nsolved={result.solved} nodes_expanded={result.nodes_expanded} "
    f"llm_calls={result.llm_calls} tokens={result.usage['total_tokens']}"
)
