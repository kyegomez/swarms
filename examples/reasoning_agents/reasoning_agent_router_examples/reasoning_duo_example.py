from swarms.agents.reasoning_agent_router import ReasoningAgentRouter

router = ReasoningAgentRouter(
    swarm_type="reasoning-duo",
    model_name="gpt-4o-mini",
    max_loops=1,
)

result = router.run("What is 2+2?")
