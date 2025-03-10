from swarms.agents.reasoning_agents import ReasoningAgentRouter

reasoning_agent_router = ReasoningAgentRouter(
    agent_name="reasoning-agent",
    description="A reasoning agent that can answer questions and help with tasks.",
    model_name="gpt-4o-mini",
    system_prompt="You are a helpful assistant that can answer questions and help with tasks.",
    max_loops=1,
    swarm_type="self-consistency",
    num_samples=1,
    output_type="list",
)

reasoning_agent_router.run(
    "What is the best possible financial strategy to maximize returns but minimize risk? Give a list of etfs to invest in and the percentage of the portfolio to allocate to each etf."
)


# reasoning_agent_router.batched_run(
#     [
#         "What is the best possible financial strategy to maximize returns but minimize risk? Give a list of etfs to invest in and the percentage of the portfolio to allocate to each etf.",
#         "What is the best possible financial strategy to maximize returns but minimize risk? Give a list of etfs to invest in and the percentage of the portfolio to allocate to each etf.",
#     ]
# )
