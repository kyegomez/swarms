from swarms.agents.reasoning_agent_router import ReasoningAgentRouter

router = ReasoningAgentRouter(
    swarm_type="self-consistency",
    model_name="gpt-4o-mini",
    max_loops=1,
    num_samples=3,
)

result = router.run("What is the capital of France?")
