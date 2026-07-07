"""
Multiple Skills Example

This example shows how an agent can use multiple skills and automatically
select the right one based on the task.
"""

from pathlib import Path
from swarms import Agent

# Get path to example_skills directory
repo_root = Path(__file__).parent.parent.parent
skills_path = repo_root / "example_skills"

# Create an agent with access to multiple skills
# The example_skills directory contains:
# - financial-analysis
# - code-review
# - data-visualization
agent = Agent(
    agent_name="Multi-Skilled Agent",
    model_name="gpt-4o",
    max_loops=1,
    skills_dir=str(skills_path),
)

print("=== Example 1: Financial Analysis Task ===")
response1 = agent.run("What are the key steps in a DCF valuation?")
print(response1[:300] + "...\n")

print("=== Example 2: Code Review Task ===")
response2 = agent.run(
    "Review this code for security issues: def login(username, password): query = f'SELECT * FROM users WHERE name={username}'"
)
print(response2[:300] + "...\n")

print("=== Example 3: Data Visualization Task ===")
response3 = agent.run(
    "What's the best chart type for showing quarterly sales trends over 3 years?"
)
print(response3[:300] + "...\n")

# Check what skills are loaded
print("\n=== Loaded Skills ===")
for skill in agent.skills_metadata:
    print(f"- {skill['name']}: {skill['description'][:60]}...")
