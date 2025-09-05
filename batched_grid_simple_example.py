from swarms import Agent
from swarms.structs.batched_grid_workflow import BatchedGridWorkflow

# Initialize the ETF-focused agent
agent = Agent(
    agent_name="ETF-Research-Agent",
    agent_description="Specialized agent for researching, analyzing, and recommending Exchange-Traded Funds (ETFs) across various sectors and markets.",
    model_name="claude-sonnet-4-20250514",
    dynamic_temperature_enabled=True,
    max_loops=1,
    dynamic_context_window=True,
)


# Create workflow with default settings
workflow = BatchedGridWorkflow(agents=[agent, agent])

# Define simple tasks
tasks = [
    "What are the best GOLD ETFs?",
    "What are the best american energy ETFs?",
]

# Run the workflow
result = workflow.run(tasks)


print(result)
