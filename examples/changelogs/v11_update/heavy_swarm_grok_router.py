from swarms import HeavySwarm

swarm = HeavySwarm(
    model_name="xai/grok-4-0709",
    question="Analyze the thermodynamic feasibility of fusion at room temperature",
)
result = swarm.run()
print(result)
