"""
DynamicSkillsLoader Integration Example

This example demonstrates the DynamicSkillsLoader integration in the Agent class.
The DynamicSkillsLoader automatically loads relevant skills based on task similarity,
allowing agents to dynamically adapt their capabilities without manual configuration.

Key features demonstrated:
- Automatic skill loading based on task requirements
- Memory efficiency by loading only necessary skills
- Seamless integration with Agent class
- Skill similarity matching using embeddings
"""

from swarms import Agent

# Create an agent with DynamicSkillsLoader integration
# The agent will automatically load skills based on task similarity
agent = Agent(
    agent_name="Adaptive Research Agent",
    model_name="gpt-4o-mini",
    max_loops=1,
    skills_dir="./examples/single_agent/agent_skill_examples",  # Directory with skill folders
)

# Task that should trigger loading of research/analysis skills
research_task = """
Conduct a comprehensive market analysis for renewable energy investments.
Include financial projections, risk assessment, and competitive landscape analysis.
Focus on solar and wind energy sectors with ROI calculations.
"""

response = agent.run(research_task)
print(response)
