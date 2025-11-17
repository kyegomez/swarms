from swarms.agents.reasoning_agents import ReasoningAgentRouter

router = ReasoningAgentRouter(
    swarm_type="GKPAgent",
    model_name="gpt-4o-mini",
    num_knowledge_items=3,
)

result = router.run("What is artificial intelligence?")
