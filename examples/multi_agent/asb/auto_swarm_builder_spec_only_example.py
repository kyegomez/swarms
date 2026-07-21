from swarms import AutoSwarmBuilder

swarm = AutoSwarmBuilder(
    name="SpecOnlyBuilder",
    description="Designs a swarm spec without executing it",
)

out = swarm.run(
    "Research the current state of solid-state batteries and "
    "write a two-paragraph summary."
)

print(out)
