from swarms import Agent, HierarchicalSwarm


def main() -> None:
    """Run a director with two specialized workers.

    Returns:
        None.
    """
    researcher = Agent(
        agent_name="Researcher",
        agent_description="Finds relevant facts and key considerations.",
        system_prompt="Research the assigned topic and return concise facts.",
        model_name="gpt-5.4",
        max_loops=1,
    )
    writer = Agent(
        agent_name="Writer",
        agent_description="Turns research into a clear final explanation.",
        system_prompt="Write a concise explanation for a general audience.",
        model_name="gpt-5.4",
        max_loops=1,
    )

    swarm = HierarchicalSwarm(
        name="ResearchWritingSwarm",
        description="Researches a topic and explains the findings.",
        agents=[researcher, writer],
        max_loops=1,
    )

    result = swarm.run(
        "Explain three practical benefits of multi-agent systems."
    )
    print(result)


if __name__ == "__main__":
    main()
