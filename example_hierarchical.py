"""
Hierarchical Multi-Agent Example

This example demonstrates hierarchical execution where a boss agent delegates tasks
to specialized worker agents, manages their execution, and synthesizes results.

Use Case: Project management where a director assigns tasks to specialized teams
"""

from swarms import Agent

# Create the boss/director agent
director_agent = Agent(
    agent_name="Director-Agent",
    agent_description="Senior director who delegates and coordinates team efforts",
    model_name="gpt-4o-mini",
    max_loops=1,
    system_prompt="You are a senior director. Break down complex projects into specific tasks and provide clear instructions to specialized teams.",
)

# Create specialized worker agents
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

# Executive agent for final approval
executive_agent = Agent(
    agent_name="Executive-Agent",
    agent_description="C-level executive who reviews and approves final plans",
    model_name="gpt-4o-mini",
    max_loops=1,
    system_prompt="You are a C-level executive. Review team outputs and provide strategic approval or feedback.",
)


def hierarchical_workflow(project_description: str):
    """Execute a hierarchical workflow with delegation."""
    
    print("="*80)
    print("HIERARCHICAL AGENT WORKFLOW")
    print("="*80)
    print(f"\nProject: {project_description}\n")
    
    # Level 1: Director breaks down the project
    print("[LEVEL 1] Director planning and delegation...")
    print("-"*80)
    director_plan = director_agent.run(
        f"Break down this project into specific tasks for Research, Development, and Marketing teams:\n\n{project_description}"
    )
    print("Director's Plan:")
    print(director_plan)
    print()
    
    # Level 2: Worker teams execute their tasks
    print("\n[LEVEL 2] Teams executing assigned tasks...")
    print("-"*80)
    
    # Research Team
    print("\n[Research Team] Working...")
    research_output = research_team.run(
        f"Based on the director's plan, conduct research:\n\n{director_plan}\n\nFocus on the research aspects."
    )
    print(f"Research completed: {len(research_output)} characters")
    
    # Development Team
    print("\n[Development Team] Working...")
    dev_output = development_team.run(
        f"Based on the director's plan, create technical specifications:\n\n{director_plan}\n\nFocus on development aspects."
    )
    print(f"Development plan completed: {len(dev_output)} characters")
    
    # Marketing Team
    print("\n[Marketing Team] Working...")
    marketing_output = marketing_team.run(
        f"Based on the director's plan, create marketing strategy:\n\n{director_plan}\n\nFocus on marketing aspects."
    )
    print(f"Marketing strategy completed: {len(marketing_output)} characters")
    
    # Level 3: Executive review and approval
    print("\n[LEVEL 3] Executive review and approval...")
    print("-"*80)
    
    combined_outputs = f"""
PROJECT PLAN:
{director_plan}

RESEARCH TEAM OUTPUT:
{research_output}

DEVELOPMENT TEAM OUTPUT:
{dev_output}

MARKETING TEAM OUTPUT:
{marketing_output}
"""
    
    executive_review = executive_agent.run(
        f"Review this complete project plan and all team outputs. Provide executive approval, feedback, and strategic recommendations:\n\n{combined_outputs}"
    )
    
    return {
        "director_plan": director_plan,
        "research": research_output,
        "development": dev_output,
        "marketing": marketing_output,
        "executive_review": executive_review
    }


if __name__ == "__main__":
    project = """
    Launch a new AI-powered mobile app for personal finance management.
    The app should help users track expenses, set budgets, and get AI-driven
    financial advice. Target launch is Q2 2024.
    """
    
    # Execute hierarchical workflow
    results = hierarchical_workflow(project)
    
    # Display final results
    print("\n" + "="*80)
    print("HIERARCHICAL WORKFLOW RESULTS")
    print("="*80)
    
    print("\n[DIRECTOR'S PLAN]")
    print("-"*80)
    print(results["director_plan"])
    
    print("\n[RESEARCH FINDINGS]")
    print("-"*80)
    print(results["research"][:300] + "..." if len(results["research"]) > 300 else results["research"])
    
    print("\n[DEVELOPMENT SPECIFICATIONS]")
    print("-"*80)
    print(results["development"][:300] + "..." if len(results["development"]) > 300 else results["development"])
    
    print("\n[MARKETING STRATEGY]")
    print("-"*80)
    print(results["marketing"][:300] + "..." if len(results["marketing"]) > 300 else results["marketing"])
    
    print("\n[EXECUTIVE REVIEW & APPROVAL]")
    print("="*80)
    print(results["executive_review"])
    
    print("\n" + "="*80)
    print("Hierarchical workflow completed!")
    print("="*80)
