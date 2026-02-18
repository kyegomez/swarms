from swarms.agents.reasoning_agent_router import ReasoningAgentRouter

# Initialize the reasoning agent router with self-consistency
reasoning_agent_router = ReasoningAgentRouter(
    agent_name="reasoning-agent",
    description="A reasoning agent that can answer questions and help with tasks.",
    model_name="gpt-4o-mini",
    system_prompt="You are a helpful assistant that can answer questions and help with tasks.",
    max_loops=1,
    swarm_type="self-consistency",
    num_samples=3,  # Generate 3 independent responses
    eval=False,  # Disable evaluation mode
    random_models_on=False,  # Disable random model selection
    majority_voting_prompt=None,  # Use default majority voting prompt
)

# Run the agent on a financial analysis task
result = reasoning_agent_router.run(
    "What is the best possible financial strategy to maximize returns but minimize risk? Give a list of etfs to invest in and the percentage of the portfolio to allocate to each etf."
)

print("Financial Strategy Result:")
print(result)
