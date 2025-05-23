from swarms.structs.agent import Agent
from swarms.structs.council_judge import CouncilAsAJudge

# ========== USAGE EXAMPLE ==========

if __name__ == "__main__":
    user_query = "How can I establish a ROTH IRA to buy stocks and get a tax break? What are the criteria?"

    base_agent = Agent(
        agent_name="Financial-Analysis-Agent",
        system_prompt="You are a financial expert helping users understand and establish ROTH IRAs.",
        model_name="gpt-4o-mini",
        max_loops=1,
    )

    model_output = base_agent.run(user_query)

    panel = CouncilAsAJudge()
    results = panel.run(user_query, model_output)

    print(results)
