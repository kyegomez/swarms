from swarms import Agent
from swarms.structs.groupchat import GroupChat


if __name__ == "__main__":
    # Initialize agents
    financial_advisor = Agent(
        agent_name="FinancialAdvisor",
        system_prompt="You are a financial advisor specializing in investment strategies and portfolio management.",
        random_models_on=True,
        output_type="final",
    )

    tax_expert = Agent(
        agent_name="TaxExpert",
        system_prompt="You are a tax expert who provides guidance on tax optimization and compliance.",
        random_models_on=True,
        output_type="final",
    )

    investment_analyst = Agent(
        agent_name="InvestmentAnalyst",
        system_prompt="You are an investment analyst focusing on market trends and investment opportunities.",
        random_models_on=True,
        output_type="final",
    )

    # Create list of agents including both Agent instances and callable
    agents = [
        financial_advisor,
        tax_expert,
        investment_analyst,
    ]

    # Initialize another chat instance in interactive mode
    interactive_chat = GroupChat(
        name="Interactive Financial Advisory Team",
        description="An interactive team of financial experts providing comprehensive financial advice",
        agents=agents,
        max_loops=1,
        output_type="all",
        interactive=True,
    )

    try:
        # Start the interactive session
        print("\nStarting interactive session...")
        # interactive_chat.run("What is the best methodology to accumulate gold and silver commodities, what is the best long term strategy to accumulate them?")
        interactive_chat.start_interactive_session()
    except Exception as e:
        print(f"An error occurred in interactive mode: {e}")
