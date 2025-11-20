from swarms import Agent


# Initialize the agent
agent = Agent(
    agent_name="Financial-Analysis-Agent",
    agent_description="Personal finance advisor agent",
    max_loops=1,
    output_type="final",
    mcp_url="http://0.0.0.0:8000/mcp",
)

out = agent.run(
    "Use the multiply tool to multiply 3 and 4 together. Look at the tools available to you.",
)

print(agent.short_memory.get_str())
