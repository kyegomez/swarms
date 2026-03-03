from swarms.structs.agent import Agent

# Initialize the agent
agent = Agent(
    agent_name="Quantitative-Trading-Agent",
    agent_description="Advanced quantitative trading and algorithmic analysis agent",
    system_prompt="You are a helpful assistant that can answer questions and help with tasks and your name is Quantitative-Trading-Agent",
    model_name="gpt-4.1",
    dynamic_temperature_enabled=True,
    max_loops="auto",
    dynamic_context_window=True,
    interactive=True,
    top_p=None,
)

out = agent.run()

print(out)
