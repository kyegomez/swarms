from swarms import Agent, one_to_one

analyst = Agent(
    agent_name="Analyst",
    system_prompt="You summarise a company's position in two sentences.",
    model_name="claude-sonnet-4-6",
    max_loops=1,
    print_on=False,
)

skeptic = Agent(
    agent_name="Skeptic",
    system_prompt="You point out the single biggest risk in the analyst's summary.",
    model_name="claude-sonnet-4-6",
    max_loops=1,
    print_on=False,
)

result = one_to_one(
    sender=analyst,
    receiver=skeptic,
    task="Assess a mid-size airline expanding into long-haul routes.",
    max_loops=1,
    output_type="str",
)

print(result)
