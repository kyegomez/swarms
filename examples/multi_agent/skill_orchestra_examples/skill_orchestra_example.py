"""
SkillOrchestra Example

Demonstrates skill-aware agent routing where tasks are decomposed into
fine-grained skills, agents are scored on competence and cost, and the
best-matched agent is selected for execution.

Based on the paper: "SkillOrchestra: Learning to Route Agents via Skill Transfer"
https://arxiv.org/abs/2602.19672
"""

from swarms.structs.agent import Agent
from swarms.structs.skill_orchestra import SkillOrchestra

# --- Define agents with distinct specializations ---

code_agent = Agent(
    agent_name="CodeExpert",
    description="Expert Python developer who writes clean, efficient, production-ready code",
    system_prompt=(
        "You are an expert Python developer. Write clean, well-documented, "
        "production-ready code with proper error handling and type hints."
    ),
    model_name="gpt-4o-mini",
    max_loops=1,
)

writer_agent = Agent(
    agent_name="TechWriter",
    description="Technical writing specialist who creates clear documentation and tutorials",
    system_prompt=(
        "You are a technical writing specialist. Write clear, comprehensive "
        "documentation with examples, explanations, and proper formatting."
    ),
    model_name="gpt-4o-mini",
    max_loops=1,
)

researcher_agent = Agent(
    agent_name="Researcher",
    description="Research analyst who gathers, synthesizes, and compares information",
    system_prompt=(
        "You are a research analyst. Provide thorough, well-structured analysis "
        "with comparisons, trade-offs, and actionable recommendations."
    ),
    model_name="gpt-4o-mini",
    max_loops=1,
)

# --- Create SkillOrchestra (auto-generates skill handbook from agent descriptions) ---

orchestra = SkillOrchestra(
    name="DevTeamOrchestra",
    agents=[code_agent, writer_agent, researcher_agent],
    model="gpt-4o-mini",
    top_k_agents=1,         # Select the single best agent per task
    learning_enabled=False,  # Set True to update skill profiles after execution
    output_type="final",     # Return only the final agent output
)

# The handbook was auto-generated. Inspect it:
print("Generated Skill Handbook:")
for skill in orchestra.skill_handbook.skills:
    print(f"  - {skill.name}: {skill.description}")
print()

# --- Run tasks: each should route to the most competent agent ---

# This should route to CodeExpert
result = orchestra.run("Write a Python function to parse and validate JSON config files")
print("=" * 60)
print("Result:")
print(result[:500] if isinstance(result, str) else str(result)[:500])
