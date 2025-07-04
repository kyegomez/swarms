import os
from dotenv import load_dotenv
from swarm_models import OpenAIChat
from swarms import Agent, GroupChat

if __name__ == "__main__":
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")

    model = OpenAIChat(
        openai_api_base="https://api.groq.com/openai/v1",
        openai_api_key=api_key,
        model_name="llama-3.1-70b-versatile",
        temperature=0.1,
    )

    # General Crypto Tax Strategist
    agent1 = Agent(
        agent_name="Token-Tax-Strategist",
        system_prompt="""You are a cryptocurrency tax specialist focusing on token trading in Florida. Your expertise includes:
        - Token-to-token swap tax implications
        - Meme coin trading tax strategies
        - Short-term vs long-term capital gains for tokens
        - Florida tax benefits for crypto traders
        - Multiple wallet tax tracking
        - High-frequency trading tax implications
        - Cost basis calculation methods for token swaps
        Provide practical tax strategies for active token traders in Florida.""",
        llm=model,
        max_loops=1,
        dynamic_temperature_enabled=True,
        user_name="swarms_corp",
        output_type="string",
        streaming_on=True,
    )

    # Compliance and Reporting Agent
    agent2 = Agent(
        agent_name="Florida-Compliance-Expert",
        system_prompt="""You are a Florida-based crypto tax compliance expert specializing in:
        - Form 8949 preparation for high-volume token trades
        - Schedule D reporting for memecoins
        - Tax loss harvesting for volatile tokens
        - Proper documentation for DEX transactions
        - Reporting requirements for airdrops and forks
        - Multi-exchange transaction reporting
        - Wash sale considerations for tokens
        Focus on compliance strategies for active memecoin and token traders.""",
        llm=model,
        max_loops=1,
        dynamic_temperature_enabled=True,
        user_name="swarms_corp",
        output_type="string",
        streaming_on=True,
    )

    # DeFi and DEX Specialist
    agent3 = Agent(
        agent_name="DeFi-Tax-Specialist",
        system_prompt="""You are a DeFi tax expert focusing on:
        - DEX trading tax implications
        - Liquidity pool tax treatment
        - Token bridging tax considerations
        - Gas fee deduction strategies
        - Failed transaction tax handling
        - Cross-chain transaction reporting
        - Impermanent loss tax treatment
        - Flash loan tax implications
        Specialize in DeFi platform tax optimization for Florida traders.""",
        llm=model,
        max_loops=1,
        dynamic_temperature_enabled=True,
        user_name="swarms_corp",
        output_type="string",
        streaming_on=True,
    )

    # Memecoin and Token Analysis Agent
    agent4 = Agent(
        agent_name="Memecoin-Analysis-Expert",
        system_prompt="""You are a memecoin and token tax analysis expert specializing in:
        - Memecoin volatility tax implications
        - Airdrop and token distribution tax treatment
        - Social token tax considerations
        - Reflective token tax handling
        - Rebase token tax implications
        - Token burn tax treatment
        - Worthless token write-offs
        - Pre-sale and fair launch tax strategies
        Provide expert guidance on memecoin and new token tax scenarios.""",
        llm=model,
        max_loops=1,
        dynamic_temperature_enabled=True,
        user_name="swarms_corp",
        output_type="string",
        streaming_on=True,
    )

    agents = [agent1, agent2, agent3, agent4]

    chat = GroupChat(
        name="Florida Token Tax Advisory",
        description="Specialized group for memecoin and token tax analysis, compliance, and DeFi trading in Florida",
        agents=agents,
    )

    # Example query focused on memecoin trading
    history = chat.run(
        "I'm trading memecoins and tokens on various DEXs from Florida. How should I handle my taxes for multiple token swaps, failed transactions, and potential losses? I have made alot of money and paid team members, delaware c corp, using crypto to pay my team"
    )
    print(history.model_dump_json(indent=2))
