from swarms.structs.agent import Agent

agent = Agent(
    agent_name="Financial-Analysis-Agent",
    agent_description="Personal finance advisor agent",
    max_loops=4,
    model_name="cerebras/llama3-70b-instruct",
    dynamic_temperature_enabled=True,
    interactive=False,
    output_type="all",
)

agent.run("Conduct an analysis of the best real undervalued ETFs")
