from swarms.structs.hiearchical_swarm import HierarchicalSwarm
from swarms.structs.agent import Agent

# Create specialized agents
research_agent = Agent(
    agent_name="Research-Analyst",
    agent_description="Specialized in comprehensive research and data gathering",
    model_name="gpt-4o-mini",
    max_loops=1,
    verbose=False,
)

analysis_agent = Agent(
    agent_name="Data-Analyst",
    agent_description="Expert in data analysis and pattern recognition",
    model_name="gpt-4o-mini",
    max_loops=1,
    verbose=False,
)

strategy_agent = Agent(
    agent_name="Strategy-Consultant",
    agent_description="Specialized in strategic planning and recommendations",
    model_name="gpt-4o-mini",
    max_loops=1,
    verbose=False,
)

# Create hierarchical swarm with interactive dashboard
swarm = HierarchicalSwarm(
    name="Swarms Corporation Operations",
    description="Enterprise-grade hierarchical swarm for complex task execution",
    agents=[research_agent, analysis_agent, strategy_agent],
    max_loops=1,
    interactive=False,  # Enable the Arasaka dashboard
    director_model_name="claude-haiku-4-5",
    director_temperature=0.7,
    director_top_p=None,
    planning_enabled=True,
)

out = swarm.run(
    "Conduct a research analysis on water stocks and etfs"
)
print(out)
