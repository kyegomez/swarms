from swarms import HeavySwarm

swarm = HeavySwarm(
    model_name="xai/grok-4-0709",
    num_agents=4,
    question="What are the key mechanisms by which GLP-1 agonists reduce cardiovascular risk?",
    verbose=True,
)
result = swarm.run()
print(result)
