from swarms import HeavySwarm


def main():
    """
    Run a HeavySwarm query to find the best 3 gold ETFs.

    This function initializes a HeavySwarm instance and queries it to provide
    the top 3 gold exchange-traded funds (ETFs), requesting clear, structured results.
    """
    swarm = HeavySwarm(
        name="Gold ETF Research Team",
        description="A team of agents that research the best gold ETFs",
        worker_model_name="claude-sonnet-4-latest",
        show_dashboard=True,
        question_agent_model_name="gpt-4.1",
        loops_per_agent=1,
    )

    prompt = (
        "Find the best 3 gold ETFs. For each ETF, provide the ticker symbol, "
        "full name, current price, expense ratio, assets under management, and "
        "a brief explanation of why it is considered among the best. Present the information "
        "in a clear, structured format suitable for investors."
    )

    out = swarm.run(prompt)
    print(out)


if __name__ == "__main__":
    main()
