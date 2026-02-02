"""
Hierarchical Multi-Agent Example

This example demonstrates hierarchical execution using HierarchicalSwarm,
where a director agent coordinates worker agents: creates plans, assigns tasks,
and iterates on results.

Use Case: Project execution where a director delegates to specialized teams.
"""

from swarms import Agent
from swarms.structs import HierarchicalSwarm

# Director agent: plans and coordinates worker agents
director = Agent(
    agent_name="Director-Agent",
    agent_description="Senior director who delegates and coordinates team efforts",
    model_name="gpt-4o-mini",
    max_loops=1,
    system_prompt="You are a senior director. Break down complex projects into specific tasks and provide clear instructions to specialized teams.",
)

# Worker agents
research_team = Agent(
    agent_name="Research-Team",
    agent_description="Research specialists who gather and analyze information",
    model_name="gpt-4o-mini",
    max_loops=1,
    system_prompt="You are a research team. Conduct thorough research on assigned topics and provide detailed findings.",
)

development_team = Agent(
    agent_name="Development-Team",
    agent_description="Development specialists who create technical solutions",
    model_name="gpt-4o-mini",
    max_loops=1,
    system_prompt="You are a development team. Design and plan technical implementations based on requirements.",
)

marketing_team = Agent(
    agent_name="Marketing-Team",
    agent_description="Marketing specialists who create promotional strategies",
    model_name="gpt-4o-mini",
    max_loops=1,
    system_prompt="You are a marketing team. Create comprehensive marketing strategies and campaigns.",
)

# Build the hierarchical swarm
swarm = HierarchicalSwarm(
    name="Project-Swarm",
    description="Director coordinates Research, Development, and Marketing teams.",
    director=director,
    agents=[research_team, development_team, marketing_team],
    max_loops=2,
    autosave=True,
    verbose=False,
)

output = swarm.run(
    "Launch a new AI-powered mobile app for personal finance management. The app should help users track expenses, set budgets, and get AI-driven financial advice. Target launch is Q2 2024."
)
print(output)
