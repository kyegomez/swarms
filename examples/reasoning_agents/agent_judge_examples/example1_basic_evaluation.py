from swarms.agents.agent_judge import AgentJudge

# Initialize the agent judge
judge = AgentJudge(
    agent_name="quality-judge", model_name="gpt-4", max_loops=2
)

# Example agent output to evaluate
agent_output = "The capital of France is Paris. The city is known for its famous Eiffel Tower and delicious croissants. The population is approximately 2.1 million people."

# Run evaluation with context building
evaluations = judge.run(task=agent_output)
