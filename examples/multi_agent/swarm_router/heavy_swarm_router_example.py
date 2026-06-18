from swarms import SwarmRouter

router = SwarmRouter(
    name="HeavySwarmRouter",
    description="A router that can route messages to the appropriate swarm",
    max_loops=1,
    swarm_type="HeavySwarm",
    heavy_swarm_max_loops=1,
    heavy_swarm_question_agent_model_name="gpt-5.4",
    heavy_swarm_worker_model_name="gpt-5.4",
)

router.run("What are the best ETFs for the american energy markets?")
