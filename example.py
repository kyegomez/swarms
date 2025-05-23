from swarms.structs.agent import Agent

# Initialize the agent
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    agent_description="Personal finance advisor agent",
    system_prompt="You are a personal finance advisor agent",
    max_loops=2,
    model_name="gpt-4o-mini",
    dynamic_temperature_enabled=True,
    interactive=True,
    output_type="all",
    safety_prompt_on=True,
)

print(agent.run("what are the rules you follow?"))
