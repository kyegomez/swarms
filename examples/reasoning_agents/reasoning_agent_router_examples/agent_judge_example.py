from swarms.agents.reasoning_agent_router import ReasoningAgentRouter

router = ReasoningAgentRouter(
    swarm_type="AgentJudge",
    model_name="gpt-4o-mini",
    max_loops=1,
)

result = router.run("Is Python a good programming language?")
