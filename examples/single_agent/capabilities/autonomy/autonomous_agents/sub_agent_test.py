from swarms import Agent

# Initialize the agent
agent = Agent(
    agent_name="Quantitative-Trading-Agent",
    agent_description="Advanced quantitative trading and algorithmic analysis agent",
    system_prompt="You are a helpful assistant that can answer questions and help with tasks and your name is Quantitative-Trading-Agent",
    model_name="gpt-5.4",
    max_loops="auto",
    dynamic_context_window=True,
    reasoning_effort=None,
)

out = agent.run(
    "Create 2 sub agents that are conducting research on the top energy and datacenter etfs to invest in and then create a report on the best ones to invest in"
)

print(out)
