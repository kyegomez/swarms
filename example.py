from swarms.structs.agent import Agent

# Initialize the agent
agent = Agent(
    agent_name="Quantitative-Trading-Agent",
    agent_description="Advanced quantitative trading and algorithmic analysis agent",
    system_prompt="You are a helpful assistant that can answer questions and help with tasks and your name is Quantitative-Trading-Agent",
    model_name="anthropic/claude-sonnet-4-5",
    dynamic_temperature_enabled=True,
    max_loops=2,
    dynamic_context_window=True,
    streaming_on=True,
    interactive=True,
    top_p=None,
)

out = agent.run(task="what is your name?")

print(out)
