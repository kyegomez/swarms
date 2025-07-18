from swarms import Agent

agent = Agent(
    name="Research Agent",
    description="A research agent that can answer questions",
    model_name="groq/moonshotai/kimi-k2-instruct",
    verbose=True,
    streaming_on=True,
    max_loops=2,
    interactive=True,
)

out = agent.run(
    "What are the best AI wechat groups in hangzhou and beijing? give me the links"
)

print(out)
