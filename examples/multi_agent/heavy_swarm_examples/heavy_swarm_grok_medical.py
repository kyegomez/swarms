"""
Grok 4.20 Heavy mode — medical research example.

Harper gathers clinical evidence, Benjamin verifies
statistical claims, Lucas challenges treatment assumptions.
Captain Swarm mediates and synthesizes.
"""

from swarms import HeavySwarm

swarm = HeavySwarm(
    name="Medical Research Team",
    description="Grok-style medical analysis",
    worker_model_name="gpt-4.1",
    question_agent_model_name="gpt-4.1",
    variant="medium",
    show_dashboard=True,
    loops_per_agent=2,
    timeout=600,
)

result = swarm.run(
    "Analyze the latest research on GLP-1 receptor agonists "
    "for treating obesity: efficacy data, long-term safety "
    "profile, cost-effectiveness vs bariatric surgery, and "
    "potential off-label applications. Include any emerging "
    "concerns about muscle mass loss and cardiovascular effects."
)

print(result)
