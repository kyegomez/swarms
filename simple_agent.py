from swarms import Agent

agent = Agent(
    name="Research Agent",
    description="A research agent that can answer questions",
    model_name="groq/moonshotai/kimi-k2-instruct",
    verbose=True,
    streaming_on=True,
)

out = agent.run(
    "Create a detailed report on the best AI research papers for multi-agent collaboration. "
    "Include paper titles, authors, publication venues, years, and a brief summary of each paper's key contributions. "
    "Highlight recent advances and foundational works. Only include papers from 2024 and 2025."
    "Provie their links as well"
)

print(out)
