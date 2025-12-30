from swarms import Agent
from swarms.structs.spreadsheet_swarm import SpreadSheetSwarm

# Example 1: Using pre-configured agents
agents = [
    Agent(
        agent_name="Research-Agent",
        agent_description="Specialized in market research and analysis",
        model_name="claude-sonnet-4-20250514",
        dynamic_temperature_enabled=True,
        max_loops=1,
        streaming_on=False,
    ),
    Agent(
        agent_name="Technical-Agent",
        agent_description="Expert in technical analysis and trading strategies",
        model_name="claude-sonnet-4-20250514",
        dynamic_temperature_enabled=True,
        max_loops=1,
        streaming_on=False,
    ),
    Agent(
        agent_name="Risk-Agent",
        agent_description="Focused on risk assessment and portfolio management",
        model_name="claude-sonnet-4-20250514",
        dynamic_temperature_enabled=True,
        max_loops=1,
        streaming_on=False,
    ),
]

# Initialize the SpreadSheetSwarm with agents
swarm = SpreadSheetSwarm(
    name="Financial-Analysis-Swarm",
    description="A swarm of specialized financial analysis agents",
    agents=agents,
    max_loops=1,
    autosave=False,
)

# Run all agents with the same task
task = "What are the top 3 energy stocks to invest in for 2024? Provide detailed analysis."
result = swarm.run(task=task)

print(result)
