from swarms import Agent

# Initialize the agent
agent = Agent(
    agent_name="Quantitative-Trading-Agent",
    agent_description="Advanced quantitative trading and algorithmic analysis agent",
    model_name="gpt-4.1",
    dynamic_temperature_enabled=True,
    max_loops=5,
    dynamic_context_window=True,
    top_p=None,
    streaming_on=True,
    interactive=True,
)

out = agent.run(
    task="What are the top five best energy stocks across nuclear, solar, gas, and other energy sources?",
)

print(out)
