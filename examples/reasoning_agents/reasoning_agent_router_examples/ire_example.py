from swarms.agents.reasoning_agent_router import ReasoningAgentRouter

router = ReasoningAgentRouter(
    swarm_type="ire",
    model_name="gpt-4o-mini",
    num_samples=1,
)

result = router.run("Explain photosynthesis in one sentence.")
print(result)
