from swarms import Agent, one_to_three

MODEL = "openrouter/deepseek/deepseek-v4-flash"

author = Agent(
    agent_name="Author",
    system_prompt="You write the opening paragraph of a short story.",
    model_name=MODEL,
    max_loops=1,
    max_tokens=2000,
    print_on=False,
)

readers = [
    Agent(
        agent_name=name,
        system_prompt=f"You are a {name}. In one sentence, say what you would change about the paragraph.",
        model_name=MODEL,
        max_loops=1,
        max_tokens=2000,
        print_on=False,
    )
    for name in ["Line-Editor", "Genre-Reader", "Teenager"]
]

result = one_to_three(
    sender=author,
    receivers=readers,
    task="Open a story about a lighthouse keeper who receives a letter.",
    output_type="str",
)

print(result)
