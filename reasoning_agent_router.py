from swarms import ReasoningAgentRouter


calculus_router = ReasoningAgentRouter(
    agent_name="calculus-expert",
    description="A calculus problem solving agent",
    model_name="gpt-4o-mini",
    system_prompt="You are a calculus expert. Solve differentiation and integration problems methodically.",
    swarm_type="self-consistency",
    num_samples=3,  # Generate 3 samples to ensure consistency
    output_type="list",
)


# Example calculus problem
calculus_problem = "Find the derivative of f(x) = x³ln(x) - 5x²"

# Get the solution
solution = calculus_router.run(calculus_problem)
print(solution)
