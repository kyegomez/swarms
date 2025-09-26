
from swarms import HeavySwarm

swarm = HeavySwarm(
    name="Gold ETF Research Team",
    description="A team of agents that research the best gold ETFs",
    worker_model_name="claude-sonnet-4-20250514",
    show_dashboard=True,
    question_agent_model_name="gpt-4.1",
    loops_per_agent=1,
    agent_prints_on=False,
    # worker_tools=[exa_search],
    random_loops_per_agent=False,
)

prompt = (
    "Find the best 3 gold ETFs. For each ETF, provide the ticker symbol, "
    "full name, current price, expense ratio, assets under management, and "
    "a brief explanation of why it is considered among the best. Present the information "
    "in a clear, structured format suitable for investors. Scrape the data from the web. "
)

out = swarm.run(prompt)

print(out)
