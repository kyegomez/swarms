from swarms import Agent, Broadcast

MODEL = "openrouter/moonshotai/kimi-k3"

announcer = Agent(
    agent_name="Announcer",
    system_prompt="You announce a company policy change in three sentences.",
    model_name=MODEL,
    max_loops=1,
    print_on=False,
)

departments = [
    Agent(
        agent_name=name,
        system_prompt=f"You lead the {name} department. In two sentences, say how this change affects your team.",
        model_name=MODEL,
        max_loops=1,
        print_on=False,
    )
    for name in ["Engineering", "Sales", "Support", "Finance"]
]

group = Broadcast(
    sender=announcer,
    receivers=departments,
    name="Policy-Broadcast",
    description="One announcement, four departmental responses.",
    output_type="dict",
)

history = group.run(
    "Announce that the company is moving to a four-day work week."
)

for message in history:
    print(f"[{message['role']}] {message['content']}\n")
