"""
Grok mode — 4-agent example.

Captain Swarm decomposes the task, then Harper (research/facts),
Benjamin (logic/math), and Lucas (creative/contrarian) work in parallel.
Captain Swarm mediates conflicts and synthesizes the final answer.
"""

from swarms import HeavySwarm

swarm = HeavySwarm(
    name="Grok 4-Agent Team",
    description="Debate-style analysis with Grok thinking-style agents",
    worker_model_name="gpt-5.4",
    question_agent_model_name="gpt-5.4",
    variant="medium",
    show_dashboard=True,
    loops_per_agent=1,
)

result = swarm.run(
    "Should a mid-size SaaS company pursue an IPO or seek acquisition "
    "in the current market? Consider valuation multiples, market conditions, "
    "team retention implications, and long-term strategic value."
)

print(result)
