"""
Basic Grok 4.20 Heavy mode example.

Uses Captain Swarm as the orchestrator with Harper (research/facts),
Benjamin (logic/math/code), and Lucas (creative/contrarian) agents.
"""

from swarms import HeavySwarm

swarm = HeavySwarm(
    name="Grok Analysis Team",
    description="Multi-agent analysis with Grok-style agents",
    worker_model_name="gpt-4.1",
    question_agent_model_name="gpt-4.1",
    variant="medium",
    show_dashboard=True,
    loops_per_agent=1,
)

result = swarm.run(
    "Should a mid-size SaaS company pursue an IPO or "
    "seek acquisition in the current market? Consider "
    "valuation multiples, market conditions, team "
    "retention implications, and long-term strategic value."
)

print(result)
