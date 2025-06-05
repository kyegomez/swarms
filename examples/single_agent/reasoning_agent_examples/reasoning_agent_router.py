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


# from swarms import ReasoningAgentRouter


# calculus_router = ReasoningAgentRouter(
#     agent_name="calculus-expert",
#     description="A calculus problem solving agent",
#     model_name="gpt-4o-mini",
#     system_prompt="You are a calculus expert. Solve differentiation and integration problems methodically.",
#     swarm_type="self-consistency",
#     num_samples=3,  # Generate 3 samples to ensure consistency
#     output_type="list",
# )


# # Example calculus problem
# calculus_problem = "Find the derivative of f(x) = x³ln(x) - 5x²"

# # Get the solution
# solution = calculus_router.run(calculus_problem)
# print(solution)
