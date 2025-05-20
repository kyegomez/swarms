from swarms.structs.auto_swarm_builder import AutoSwarmBuilder

example = AutoSwarmBuilder(
    name="ContentCreation-Swarm",
    description="A swarm of specialized AI agents for research, writing, editing, and publishing that maintain brand consistency across channels while automating distribution.",
    max_loops=1,
)

print(
    example.run(
        "Build agents for research, writing, editing, and publishing to enhance brand consistency and automate distribution across channels."
    )
)
