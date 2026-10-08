from swarms import Agent

Agent(
    agent_name="Stock-Analysis-Agent",
    model_name="deepseek/deepseek-reasoner",
    max_loops="auto",
    interactive=True,
    streaming_on=False,
).run("What are 5 hft algorithms")
