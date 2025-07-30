from swarms import Agent

agent = Agent(
    name="Research Agent",
    description="A research agent that can answer questions",
    model_name="claude-3-5-sonnet-20241022",
    streaming_on=True,
    max_loops=1,
    interactive=True,
)

out = agent.run(
    "What are the best arbitrage trading strategies for altcoins? Give me research papers and articles on the topic."
)

print(out)
