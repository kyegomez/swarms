from swarms import Agent, OneToOne

writer = Agent(
    agent_name="Writer",
    system_prompt="You write short product taglines. Reply with one tagline.",
    model_name="gpt-5.4",
    max_loops=1,
    print_on=False,
)

editor = Agent(
    agent_name="Editor",
    system_prompt="You are a strict editor. Rewrite the tagline to be shorter and punchier.",
    model_name="gpt-5.4",
    max_loops=1,
    print_on=False,
)

pair = OneToOne(
    sender=writer,
    receiver=editor,
    name="Tagline-Pair",
    description="A writer drafts, an editor tightens.",
    output_type="dict",
)

history = pair.run(
    "Write a tagline for a password manager.", max_loops=2
)

for message in history:
    print(f"[{message['role']}] {message['content']}\n")
