"""
Grok 4.20 Heavy mode — no dashboard, minimal output.

Good for batch processing or programmatic usage.
"""

from swarms import HeavySwarm

swarm = HeavySwarm(
    name="Quiet Grok Team",
    worker_model_name="gpt-4.1",
    question_agent_model_name="gpt-4.1",
    variant="medium",
    show_dashboard=False,
    agent_prints_on=False,
    output_type="string",
)

result = swarm.run(
    "What are the top 3 risks and top 3 opportunities "
    "for commercial real estate in 2026?"
)

print(result)
