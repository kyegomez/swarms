from swarms import Agent, OneToThree

MODEL = "openrouter/qwen/qwen3.8-flash"

founder = Agent(
    agent_name="Founder",
    system_prompt="You pitch a startup idea in three sentences.",
    model_name=MODEL,
    max_loops=1,
    print_on=False,
)

investors = [
    Agent(
        agent_name=name,
        system_prompt=f"You are a {style} investor. In two sentences, react to the pitch.",
        model_name=MODEL,
        max_loops=1,
        print_on=False,
    )
    for name, style in [
        ("Growth-VC", "growth-focused"),
        ("Value-Investor", "cautious, numbers-first"),
        ("Angel", "founder-friendly"),
    ]
]

panel = OneToThree(
    sender=founder,
    receivers=investors,
    name="Pitch-Panel",
    description="One pitch, three investor reactions.",
    output_type="dict",
)

history = panel.run(
    "Pitch a marketplace for renting professional camera gear."
)

for message in history:
    print(f"[{message['role']}] {message['content']}\n")
