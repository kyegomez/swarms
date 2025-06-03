from swarms.agents.gkp_agent import GKPAgent

# Initialize the GKP Agent
agent = GKPAgent(
    agent_name="gkp-agent",
    model_name="gpt-4o-mini",  # Using OpenAI's model
    num_knowledge_items=6,  # Generate 6 knowledge items per query
)

# Example queries
queries = [
    "What are the implications of quantum entanglement on information theory?",
]

# Run the agent
results = agent.run(queries)

# Print results
for i, result in enumerate(results):
    print(f"\nQuery {i+1}: {queries[i]}")
    print(f"Answer: {result}")
