from swarms.structs.agent import Agent

agent = Agent(
    agent_name="test",
    agent_description="test",
    system_prompt="test",
)

print(agent.list_output_types())
