"""
Agent Skills Example

This example shows how to use Agent Skills - modular, reusable capabilities
defined in SKILL.md files. Skills are automatically loaded and the agent follows
their instructions when relevant.

Note: This example uses the example_skills directory in the root of the repo.
      To use your own skills, create a skills directory and point to it.
"""

from pathlib import Path
from swarms import Agent

# Get path to example_skills directory (2 levels up from this file)
repo_root = Path(__file__).parent.parent.parent
skills_path = repo_root / "example_skills"

# Create an agent with skills
agent = Agent(
    agent_name="Financial Analyst",
    model_name="gpt-4o",
    max_loops=1,
    skills_dir=str(skills_path),
)

# The agent automatically uses the financial-analysis skill for this task
response = agent.run(
    "Analyze Apple's financials and provide a DCF valuation framework"
)

print(response)
