from swarms import Agent

Agent(
    agent_name="Stock-Analysis-Agent",
    model_name="gpt-4o-mini",
    max_loops=1,
    interactive=False,
    streaming_on=True,
).run("What are 5 hft algorithms")
