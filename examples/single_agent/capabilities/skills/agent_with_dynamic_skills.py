"""
Dynamic Skills Loading Example

This example demonstrates how to use dynamic skills loading with an agent.
Instead of loading all skills statically, the agent loads only the skills that are
relevant to the specified task based on cosine similarity.

The dynamic loading happens in the `handle_skills()` method during `agent.run()`.
Skills are selected for each task from the bundled skill folders beside this script.
"""

from pathlib import Path

from swarms import Agent

# Create an agent using the bundled skill folders beside this script.
agent = Agent(
    agent_name="Dynamic Skills Agent",
    model_name="gpt-4o",
    max_loops=1,
    skills_dir=str(Path(__file__).resolve().parent),
)

# Task 1: Financial analysis task (should load financial-analysis skill)
financial_task = """
Analyze the quarterly financial performance of a tech company.
Calculate key metrics like revenue growth, profit margins, and ROI.
Create financial projections and risk assessments.
financial analysis
"""

print("=== Testing Financial Analysis Task ===")
print("Task:", financial_task.strip())
response1 = agent.run(financial_task)
print(response1)
