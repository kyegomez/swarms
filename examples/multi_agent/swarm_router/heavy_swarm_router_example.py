from swarms import SwarmRouter

router = SwarmRouter(
    name="HeavySwarmRouter",
    description="A router that can route messages to the appropriate swarm",
    max_loops=1,
    swarm_type="HeavySwarm",
    heavy_swarm_loops_per_agent=1,
    heavy_swarm_question_agent_model_name="gpt-4o",
    heavy_swarm_worker_model_name="gpt-4o",
)

router.run("What are the best ETFs for the american energy markets?")
