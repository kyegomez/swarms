from swarms.structs.auto_swarm_builder import AutoSwarmBuilder


example = AutoSwarmBuilder(
    name="ChipDesign-Swarm",
    description="A swarm of specialized AI agents collaborating on chip architecture, logic design, verification, and optimization to create novel semiconductor designs",
    max_loops=1,
)

print(
    example.run(
        "Design a new AI accelerator chip optimized for transformer model inference. Consider the following aspects: 1) Overall chip architecture and block diagram 2) Memory hierarchy and interconnects 3) Processing elements and data flow 4) Power and thermal considerations 5) Physical layout recommendations -> "
    )
)
