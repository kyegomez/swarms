"""
TWZRD Agent Intel - MCP Server Integration for Swarms
======================================================

Trust-score Solana AI agent wallets before sending x402 USDC micropayments.

Tools exposed:
  - score_agent(wallet)         : Returns a trust score (0-100) for a Solana agent wallet
  - preflight_check(wallet)     : Full readiness check (score + stake + age + activity)
  - get_trust_receipt(wallet)   : Paid HTTP 402 endpoint; returns a signed trust receipt

Zero-install MCP config:
    {"mcpServers": {"twzrd-agent-intel": {"url": "https://intel.twzrd.xyz/mcp"}}}
"""

import os
from swarms import Agent
from swarms.tools.mcp_client_call import execute_mcp_tool

TWZRD_MCP_URL = "https://intel.twzrd.xyz/mcp"


def score_agent_wallet(wallet: str) -> dict:
    """
    Score a Solana agent wallet for trust before sending a payment.

    Args:
        wallet: Solana wallet address (base58)

    Returns:
        dict with score (0-100), flags, and recommendation
    """
    return execute_mcp_tool(
        url=TWZRD_MCP_URL,
        tool_name="score_agent",
        params={"wallet": wallet},
    )


def preflight_check(wallet: str) -> dict:
    """
    Full preflight check: score + stake status + wallet age + recent activity.
    Use before initiating an x402 USDC payment to an unknown agent wallet.

    Args:
        wallet: Solana wallet address (base58)

    Returns:
        dict with score, stake, age_days, last_active, recommendation
    """
    return execute_mcp_tool(
        url=TWZRD_MCP_URL,
        tool_name="preflight_check",
        params={"wallet": wallet},
    )


# Example: Swarms agent with trust-gated tool calls
agent = Agent(
    agent_name="Trust-Gated Payment Agent",
    system_prompt=(
        "You are an autonomous payment agent. Before authorizing any USDC "
        "micropayment to a Solana wallet, you MUST call preflight_check() and "
        "confirm the trust score is above 60. If the score is below 60, abort "
        "the payment and explain why."
    ),
    tools=[score_agent_wallet, preflight_check],
    max_loops=3,
    model_name="gpt-4o-mini",
    verbose=True,
)


def main():
    target_wallet = os.getenv(
        "TARGET_AGENT_WALLET",
        "4LkEFjhCVsQGe6VQdaKBRnvzS8vfKeJH5bC2kfExRBnP",
    )
    result = agent.run(
        f"I need to send 0.10 USDC to Solana wallet {target_wallet} for "
        "intelligence data. Please verify the wallet is trustworthy before "
        "I authorize the payment."
    )
    print(result)


if __name__ == "__main__":
    main()
