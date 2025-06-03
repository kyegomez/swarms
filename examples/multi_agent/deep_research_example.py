from swarms.structs.deep_research_swarm import DeepResearchSwarm

swarm = DeepResearchSwarm(
    name="Deep Research Swarm",
    description="A swarm that conducts comprehensive research across multiple domains",
    max_loops=1,
)


swarm.run(
    "What are the biggest gas and oil companies in russia? Only provide 3 queries"
)
