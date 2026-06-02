from swarms.structs.agent import Agent
from swarms.structs.council_as_judge import CouncilAsAJudge


if __name__ == "__main__":
    user_query = "How can I establish a ROTH IRA to buy stocks and get a tax break? What are the criteria?"

    base_agent = Agent(
        agent_name="Financial-Analysis-Agent",
        system_prompt="You are a financial expert helping users understand and establish ROTH IRAs.",
        model_name="claude-opus-4-20250514",
        max_loops=1,
        max_tokens=16000,
    )

    panel = CouncilAsAJudge(base_agent=base_agent)
    results = panel.run(user_query)

    print(results)
