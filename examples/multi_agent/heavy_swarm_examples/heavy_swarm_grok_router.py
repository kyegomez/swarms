"""
Grok 4.20 Heavy mode via SwarmRouter.

Use SwarmRouter to run HeavySwarm with Grok agents.
"""

from swarms import SwarmRouter

router = SwarmRouter(
    name="GrokRouter",
    description="SwarmRouter with Grok Heavy agents",
    swarm_type="HeavySwarm",
    heavy_swarm_worker_model_name="gpt-5.4",
    heavy_swarm_question_agent_model_name="gpt-5.4",
    heavy_swarm_variant="medium",
)

result = router.run(
    "Compare the investment potential of Bitcoin, "
    "Ethereum, and Solana for a conservative portfolio "
    "in 2026. Factor in regulatory risk, technical "
    "fundamentals, and institutional adoption trends."
)

print(result)
