from swarms.structs.deep_research_swarm import DeepResearchSwarm


def main():
    swarm = DeepResearchSwarm(
        name="Deep Research Swarm",
        description="A swarm of agents that can perform deep research on a given topic",
        output_type="string",  # Change to string output type for better readability
    )

    # Format the query as a proper question
    query = "What are the latest developments and news in the AI and cryptocurrency space?"

    try:
        result = swarm.run(query)
        print("\nResearch Results:")
        print(result)
    except Exception as e:
        print(f"Error occurred: {str(e)}")


if __name__ == "__main__":
    main()
