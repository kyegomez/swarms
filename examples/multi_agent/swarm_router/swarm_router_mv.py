from swarms import SwarmRouter, Agent

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

router = SwarmRouter(
    name="SwarmRouter",
    description="Routes tasks to specialized agents based on their capabilities",
    agents=[research_agent, analysis_agent, strategy_agent],
    swarm_type="MajorityVoting",
    max_loops=1,
    verbose=False,
)

result = router.run(
    "Conduct a research analysis on water stocks and etfs"
)
print(result)
