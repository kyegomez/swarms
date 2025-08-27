from swarms import HierarchicalSwarm, Agent

# Create agents
research_agent = Agent(
    agent_name="Research-Analyst", model_name="gpt-4.1", print_on=True
)
analysis_agent = Agent(
    agent_name="Data-Analyst", model_name="gpt-4.1", print_on=True
)

# Create swarm with interactive dashboard
swarm = HierarchicalSwarm(
    agents=[research_agent, analysis_agent],
    max_loops=1,
    interactive=True,  # Enable the Arasaka dashboard
    multi_agent_prompt_improvements=True,
)

# Run swarm (task will be prompted interactively)
result = swarm.run("what are the best nanomachine research papers?")

print(result)
