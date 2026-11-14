from swarms import Agent, ContextTree

tree = ContextTree(
    spec=(
        "Plan a two-day offsite in Lisbon in May for 12 engineers "
        "flying from Berlin. The total budget is 9,000 EUR."
    )
)

parts = {
    "Venue": "Pick a venue and give its price for two days.",
    "Travel": "Plan flights and local transport, with prices.",
    "Agenda": "Write the agenda around the venue and travel times.",
}

agents = {
    name: Agent(
        agent_name=name,
        system_prompt=(
            "You do one part of a team task. Save every fact a "
            "teammate needs with context_write, one finding per call."
        ),
        model_name="gpt-5.4-mini",
        max_loops=3,
        tools=tree.tools(author=name, tree_id="offsite", depth=1),
        print_on=False,
    )
    for name in parts
}

# Each agent starts from the spec and the notes, not transcripts.
for name, part in parts.items():
    context = tree.render(tree_id="offsite", depth=1)
    agents[name].run(f"{context}\n\nYour part: {part}")

print("Notes written:")
for note in tree.notes:
    print(f"- [{note['scope']}] {note['author']}: {note['text']}")

print("\nWhat the next agent on this task would read:")
print(tree.render(tree_id="offsite", depth=1))

for name, agent in agents.items():
    print(f"\n{name} usage: {agent.usage}")
