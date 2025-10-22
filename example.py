import json

from swarms import Agent

# Initialize the agent
agent = Agent(
    agent_name="Quantitative-Trading-Agent",
    agent_description="Advanced quantitative trading and algorithmic analysis agent",
    model_name="gpt-4.1",
    dynamic_temperature_enabled=True,
    max_loops=1,
    dynamic_context_window=True,
    streaming_on=False,
    top_p=None,
    output_type="dict",
)

out = agent.run(
    task="What are the top five best energy stocks across nuclear, solar, gas, and other energy sources?",
    n=1,
)
print(json.dumps(out, indent=4))
