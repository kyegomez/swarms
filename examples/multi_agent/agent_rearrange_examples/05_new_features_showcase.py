"""Minimal demo of the four new AgentRearrange features."""

from swarms import Agent, AgentRearrange


def make_agent(name: str) -> Agent:
    return Agent(
        agent_name=name,
        system_prompt=f"You are {name}. Respond in one short sentence.",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
        persistent_memory=False,
    )


a, b, c = make_agent("A"), make_agent("B"), make_agent("C")


# 1. explain() — print the execution plan, no LLM calls.
AgentRearrange(agents=[a, b, c], flow="A -> B, C").explain()


# 2. Strict output_type validation.
try:
    AgentRearrange(agents=[a, b], flow="A -> B", output_type="dictt")
except ValueError as e:
    print(f"\nrejected bad output_type: {e}\n")


# 3. Pure-concurrent flow (no '->' required).
fanout = AgentRearrange(agents=[a, b, c], flow="A, B, C")
fanout.run("Say hello.")
print("pure fan-out OK\n")


# 4. team_awareness no longer pollutes the shared conversation.
aware = AgentRearrange(
    agents=[a, b],
    flow="A -> B -> A",
    team_awareness=True,
)
aware.run("Pick a color.")
leaks = [
    m
    for m in aware.conversation.to_dict()
    if "Sequential awareness" in str(m.get("content", ""))
]
print(f"awareness leaks in shared conversation: {len(leaks)}")
