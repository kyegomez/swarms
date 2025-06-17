from swarms import Agent, MultiAgentRouter

if __name__ == "__main__":
    agents = [
        Agent(
            agent_name="Researcher",
            system_prompt="Answer questions briefly.",
            model_name="gpt-4o-mini",
        ),
        Agent(
            agent_name="Coder",
            system_prompt="Write small Python functions.",
            model_name="gpt-4o-mini",
        ),
    ]

    router = MultiAgentRouter(agents=agents)

    result = router.route_task("Write a function that adds two numbers")
