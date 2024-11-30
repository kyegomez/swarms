from swarms import Agent

Agent(
    agent_name="Stock-Analysis-Agent",
    model_name="gpt-4o-mini",
    max_loops="auto",
    streaming_on=True,
    interactive=True,
).run("What are 5 hft algorithms")
