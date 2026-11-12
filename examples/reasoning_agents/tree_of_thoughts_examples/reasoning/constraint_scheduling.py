"""Constraint satisfaction: fit five talks into five slots under six rules.

Breadth-first search places one speaker (or rules out slots) per step and
keeps three partial schedules alive, so an early placement that later
proves impossible does not sink the search. Expected answer: Dana 9:00,
Ben 10:00, Chen 11:00, Alice 13:00, Eli 14:00.
"""

from swarms import TreeOfThoughts

agent = TreeOfThoughts(
    name="Scheduler",
    model_name="gpt-5.4",
    search_algorithm="bfs",
    generation_strategy="propose",
    evaluation_strategy="value",
    num_thoughts=3,
    breadth=3,
    max_depth=5,
    thought_description=(
        "One deduction or placement: fix one speaker's slot, or rule out "
        "slots for a speaker, naming the constraints that force it."
    ),
    evaluation_criteria=(
        "Does the partial schedule break any constraint, including one "
        "the remaining free slots can no longer satisfy? A final schedule "
        "gives every speaker a distinct slot and satisfies all six "
        "constraints."
    ),
)

answer = agent.run(
    "Schedule five talks by Alice, Ben, Chen, Dana and Eli into the slots "
    "9:00, 10:00, 11:00, 13:00 and 14:00 (lunch is at 12:00), one talk per "
    "slot. Constraints: (1) Alice cannot speak before 11:00. (2) Ben "
    "speaks in the slot immediately before Chen's. (3) Dana speaks in the "
    "morning. (4) Eli speaks after Alice. (5) Chen does not take 14:00. "
    "(6) Ben does not take 9:00. Give the full schedule."
)
print(f"Answer: {answer}\n")

result = agent.last_result
for number, step in enumerate(result.steps, 1):
    print(f"{number}. {step}")
print(
    f"\nsolved={result.solved} nodes_expanded={result.nodes_expanded} "
    f"llm_calls={result.llm_calls} tokens={result.usage['total_tokens']}"
)
