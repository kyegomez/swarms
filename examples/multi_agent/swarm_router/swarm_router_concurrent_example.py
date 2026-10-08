from swarms import Agent, SwarmRouter


def main():
    agents = [
        Agent(
            agent_name="Researcher",
            system_prompt="You provide concise research notes.",
            model_name="gpt-5.4",
            max_loops=1,
        ),
        Agent(
            agent_name="Writer",
            system_prompt="You turn ideas into clear summaries.",
            model_name="gpt-5.4",
            max_loops=1,
        ),
    ]

    router = SwarmRouter(
        name="simple-concurrent-router",
        swarm_type="ConcurrentWorkflow",
        agents=agents,
        max_loops=1,
    )

    result = router.run(
        "Give two short benefits of using multi-agent systems."
    )
    print(result)


if __name__ == "__main__":
    main()
