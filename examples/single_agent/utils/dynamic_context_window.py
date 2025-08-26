from swarms import Agent


def main():
    """
    Run a quantitative trading agent to recommend top 3 gold ETFs.
    """
    agent = Agent(
        agent_name="Quantitative-Trading-Agent",
        agent_description="Advanced quantitative trading and algorithmic analysis agent",
        system_prompt=(
            "You are an expert quantitative trading agent. "
            "Recommend the best gold ETFs using your expertise in trading strategies, "
            "risk management, and financial analysis. Be concise and precise."
        ),
        model_name="claude-sonnet-4-20250514",
        dynamic_temperature_enabled=True,
        max_loops=1,
        dynamic_context_window=True,
    )

    out = agent.run(
        task="What are the best top 3 etfs for gold coverage?"
    )
    print(out)


if __name__ == "__main__":
    main()
