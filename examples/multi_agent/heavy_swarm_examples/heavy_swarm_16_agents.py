"""
Grok Heavy mode — 16-agent example.

Grok (captain) decomposes the task into 15 domain-specific questions,
all 15 specialists work in parallel, then Grok synthesizes the outputs.
"""

from swarms import HeavySwarm

swarm = HeavySwarm(
    name="Grok Heavy 16-Agent Team",
    description="16-agent deep multi-domain analysis",
    worker_model_name="gpt-4.1",
    question_agent_model_name="gpt-4.1",
    variant="heavy",
    show_dashboard=True,
    loops_per_agent=1,
)

result = swarm.run(
    "What are the best energy stocks fueling AI growth? Analyze the top companies "
    "supplying power infrastructure, nuclear, natural gas, and renewables to data centers, "
    "including financial performance, competitive positioning, and long-term growth outlook."
)

print(result)
