from swarms import HeavySwarm

swarm = HeavySwarm(
    name="Grok Heavy Research Team",
    description="16-agent Grok 4.20 Heavy mode for deep multi-domain analysis",
    worker_model_name="grok-4",
    question_agent_model_name="grok-4",
    show_dashboard=True,
    loops_per_agent=1,
    agent_prints_on=False,
    variant="heavy",
)

prompt = (
    "What are the most transformative technologies that will reshape "
    "civilization over the next 50 years? Analyze from all relevant dimensions."
)

out = swarm.run(prompt)
print(out)
