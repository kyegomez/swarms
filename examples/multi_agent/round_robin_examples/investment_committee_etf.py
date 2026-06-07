"""
Investment-committee roundtable on whether to launch a thematic AI ETF.

Four committee members rotate in fixed order across two full loops, so each
member gets two turns. Each turn, the speaker sees the full transcript so
far plus who spoke before them and who is up next, which is what makes the
round-robin (rather than parallel) structure useful here: later speakers
can directly respond to the prior speaker's case.
"""

from swarms import Agent, RoundRobinSwarm

cio = Agent(
    agent_name="CIO",
    agent_description="Chief Investment Officer.",
    system_prompt=(
        "You are the Chief Investment Officer. Lead with the strategic "
        "case: market fit, expected AUM, and how the product slots into "
        "the existing lineup. Be concrete about target investors."
    ),
    model_name="claude-opus-4-7",
    max_loops=1,
)

risk_officer = Agent(
    agent_name="RiskOfficer",
    agent_description="Head of risk management.",
    system_prompt=(
        "You are the Head of Risk. Stress-test the prior speaker's "
        "argument: concentration risk, drawdown scenarios in an AI "
        "selloff, tracking error, liquidity of the underlying basket. "
        "Quantify wherever possible."
    ),
    model_name="gpt-5.4",
    max_loops=1,
)

allocation_strategist = Agent(
    agent_name="AllocationStrategist",
    agent_description="Portfolio construction lead.",
    system_prompt=(
        "You are the Allocation Strategist. Propose the index "
        "methodology: weighting scheme, rebalance cadence, inclusion "
        "criteria, and how it differs from existing thematic ETFs in "
        "the market."
    ),
    model_name="gemini/gemini-2.5-pro",
    max_loops=1,
)

compliance = Agent(
    agent_name="Compliance",
    agent_description="Compliance and regulatory counsel.",
    system_prompt=(
        "You are Compliance. Flag any regulatory, disclosure, or "
        "naming-rule concerns (e.g. SEC Names Rule 35d-1). State what "
        "must be true for the product to be approved."
    ),
    model_name="claude-sonnet-4-6",
    max_loops=1,
)

committee = RoundRobinSwarm(
    name="AI-ETF-Investment-Committee",
    agents=[cio, risk_officer, allocation_strategist, compliance],
    max_loops=2,
    output_type="all",
    verbose=True,
)

result = committee.run(
    "Should we launch a new thematic ETF targeting the AI infrastructure "
    "supply chain (chips, datacenter REITs, power/cooling, networking)? "
    "Reach a recommendation."
)

print(result)
