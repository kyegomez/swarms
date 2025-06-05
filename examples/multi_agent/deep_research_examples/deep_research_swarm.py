from swarms.structs.deep_research_swarm import DeepResearchSwarm


def main():
    swarm = DeepResearchSwarm(
        name="Deep Research Swarm",
        description="A swarm of agents that can perform deep research on a given topic",
    )

    swarm.run("What are the latest news in the AI an crypto space")


main()
