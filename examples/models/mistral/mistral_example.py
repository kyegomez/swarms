from swarms import Agent

# Initialize the agent
agent = Agent(
    agent_name="Quantitative-Trading-Agent",
    agent_description="Quantitative trading and analysis agent",
    system_prompt="You are an expert quantitative trading agent. Answer concisely and accurately using your knowledge of trading strategies, risk management, and financial markets.",
    model_name="mistral/mistral-tiny",
    dynamic_temperature_enabled=True,
    output_type="str-all-except-first",
    max_loops="auto",
    interactive=True,
    no_reasoning_prompt=True,
    streaming_on=True,
)

out = agent.run(
    task="What are the best top 3 etfs for gold coverage?"
)
print(out)
