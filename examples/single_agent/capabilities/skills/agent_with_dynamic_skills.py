"""
Dynamic Skills Loading Example

This example demonstrates how to use dynamic skills loading with an agent.
Instead of loading all skills statically, the agent loads only the skills that are
relevant to the specified task based on cosine similarity during initialization.

The dynamic loading happens in the `handle_skills()` method when a `task` parameter
is provided to the Agent constructor. Skills are loaded once during initialization
based on the task description.
"""

from swarms import Agent

# Step 1: Create skills directory structure (using existing skills)
# We'll use the existing skills in agent_skill_examples

# Step 2: Create agent with dynamic skills loading enabled
agent = Agent(
    agent_name="Dynamic Skills Agent",
    model_name="gpt-4o",
    max_loops=1,
    skills_dir="./agent_skill_examples",  # Directory containing skill folders
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
