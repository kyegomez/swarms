"""
AOP x402 Payment Integration Example

This example demonstrates how to enable x402 cryptocurrency payments
for all agents in an AOP cluster. Users will need to pay in cryptocurrency
before accessing any agent endpoints.

Requirements:
    pip install swarms x402

Usage:
    python aop_payment_example.py

Then test with your MCP client or directly via HTTP.
"""

from dotenv import load_dotenv

from swarms import Agent
from swarms.structs.aop import AOP, PaymentConfig

# Load environment variables
load_dotenv()


def main():
    # Create multiple agents for the cluster
    research_agent = Agent(
        agent_name="Research-Agent",
        system_prompt="You are an expert research analyst. Conduct thorough research on the given topic and provide comprehensive insights.",
        model_name="gpt-4o-mini",
        max_loops=1,
    )

    analysis_agent = Agent(
        agent_name="Analysis-Agent",
        system_prompt="You are a data analysis expert. Analyze the provided information and extract key insights.",
        model_name="gpt-4o-mini",
        max_loops=1,
    )

    writing_agent = Agent(
        agent_name="Writing-Agent",
        system_prompt="You are a professional writer. Create well-structured, engaging content based on the given requirements.",
        model_name="gpt-4o-mini",
        max_loops=1,
    )

    # Configure x402 payment settings
    # Replace with your actual Solana wallet address!
    payment_config = PaymentConfig(
        pay_to_address="YourSolanaWalletAddressHere",  # REQUIRED: Your Solana wallet
        price="$0.01",  # Price per agent request
        network_id="solana",  # Solana mainnet (use "solana-devnet" for testing)
        description="AI Agent Marketplace - Pay per use",
    )

    # Create AOP cluster with payment enabled
    aop = AOP(
        server_name="Paid AI Agent Cluster",
        description="A marketplace of AI agents with x402 cryptocurrency payments",
        agents=[research_agent, analysis_agent, writing_agent],
        port=8000,
        host="localhost",
        payment=True,  # Enable x402 payment
        payment_config=payment_config,  # Payment configuration
        verbose=True,
        log_level="INFO",
    )

    print("\n" + "=" * 60)
    print("🚀 AOP Server with x402 Payment Integration (Solana)")
    print("=" * 60)
    print(f"✅ Payment enabled: {payment_config.price} per request")
    print(f"💰 Solana wallet: {payment_config.pay_to_address}")
    print(f"🌐 Network: {payment_config.network_id}")
    print(f"📍 Server: http://{aop.host}:{aop.port}")
    print("\n💡 All MCP endpoints require Solana payment via x402!")
    print("🎯 Available Agents:")
    for agent_name in aop.list_agents():
        print(f"   - {agent_name}")
    print("\n" + "=" * 60)
    print("Starting server...\n")

    # Start the server
    aop.run()


if __name__ == "__main__":
    main()
