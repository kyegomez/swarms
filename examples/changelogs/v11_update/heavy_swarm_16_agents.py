from swarms import HeavySwarm

# 16-agent mode for maximum research depth
swarm = HeavySwarm(
    model_name="xai/grok-4-0709",
    num_agents=16,
    question="Synthesize the current state of quantum error correction and near-term scalability prospects",
)
result = swarm.run()
print(result)
