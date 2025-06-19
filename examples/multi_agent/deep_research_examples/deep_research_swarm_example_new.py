from swarms.structs.deep_research_swarm import DeepResearchSwarm


swarm = DeepResearchSwarm(
    name="Deep Research Swarm",
    description="A swarm of agents that can perform deep research on a given topic",
    output_type="all",  # Change to string output type for better readability
)

out = swarm.run(
    "What are the latest developments and news in the AI and cryptocurrency space?"
)
print(out)
