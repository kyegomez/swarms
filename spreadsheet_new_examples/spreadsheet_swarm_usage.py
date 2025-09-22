"""
SpreadSheetSwarm Usage Examples
==============================

This file demonstrates the two main ways to use SpreadSheetSwarm:
1. With pre-configured agents
2. With CSV configuration file
"""

from swarms import Agent
from swarms.structs.spreadsheet_swarm import SpreadSheetSwarm


def example_with_agents():
    """Example using pre-configured agents"""
    print("=== Example 1: Using Pre-configured Agents ===")

    # Create agents
    agents = [
        Agent(
            agent_name="Writer-Agent",
            agent_description="Creative writing specialist",
            model_name="claude-sonnet-4-20250514",
            dynamic_temperature_enabled=True,
            max_loops=1,
            streaming_on=False,
            print_on=False,
        ),
        Agent(
            agent_name="Editor-Agent",
            agent_description="Content editing and proofreading expert",
            model_name="claude-sonnet-4-20250514",
            dynamic_temperature_enabled=True,
            max_loops=1,
            streaming_on=False,
            print_on=False,
        ),
    ]

    # Create swarm with agents
    swarm = SpreadSheetSwarm(
        name="Content-Creation-Swarm",
        description="A swarm for content creation and editing",
        agents=agents,
        autosave=True,
        max_loops=1,
    )

    # Run with same task for all agents
    result = swarm.run("Write a short story about AI and creativity")

    print(f"Tasks completed: {result['tasks_completed']}")
    print(f"Number of agents: {result['number_of_agents']}")
    return result


def example_with_csv():
    """Example using CSV configuration"""
    print("\n=== Example 2: Using CSV Configuration ===")

    # Create CSV content
    csv_content = """agent_name,description,system_prompt,task,model_name,max_loops,user_name
Writer-Agent,Creative writing specialist,You are a creative writer,Write a poem about technology,claude-sonnet-4-20250514,1,user
Editor-Agent,Content editing expert,You are an editor,Review and improve the poem,claude-sonnet-4-20250514,1,user
Critic-Agent,Literary critic,You are a literary critic,Provide constructive feedback on the poem,claude-sonnet-4-20250514,1,user"""

    # Save CSV file
    with open("agents.csv", "w") as f:
        f.write(csv_content)

    # Create swarm with CSV path only (no agents provided)
    swarm = SpreadSheetSwarm(
        name="Poetry-Swarm",
        description="A swarm for poetry creation and review",
        load_path="agents.csv",  # No agents parameter - will load from CSV
        autosave=True,
        max_loops=1,
    )

    # Run with different tasks from CSV
    result = swarm.run_from_config()

    print(f"Tasks completed: {result['tasks_completed']}")
    print(f"Number of agents: {result['number_of_agents']}")

    # Clean up
    import os

    if os.path.exists("agents.csv"):
        os.remove("agents.csv")

    return result


def example_mixed_usage():
    """Example showing both agents and CSV can be used together"""
    print("\n=== Example 3: Mixed Usage (Agents + CSV) ===")

    # Create one agent
    agent = Agent(
        agent_name="Coordinator-Agent",
        agent_description="Project coordinator",
        model_name="claude-sonnet-4-20250514",
        dynamic_temperature_enabled=True,
        max_loops=1,
        streaming_on=False,
        print_on=False,
    )

    # Create CSV content
    csv_content = """agent_name,description,system_prompt,task,model_name,max_loops,user_name
Researcher-Agent,Research specialist,You are a researcher,Research the topic thoroughly,claude-sonnet-4-20250514,1,user
Analyst-Agent,Data analyst,You are a data analyst,Analyze the research data,claude-sonnet-4-20250514,1,user"""

    with open("research_agents.csv", "w") as f:
        f.write(csv_content)

    # Create swarm with both agents and CSV
    swarm = SpreadSheetSwarm(
        name="Mixed-Swarm",
        description="A swarm with both pre-configured and CSV-loaded agents",
        agents=[agent],  # Pre-configured agent
        load_path="research_agents.csv",  # CSV agents
        autosave=True,
        max_loops=1,
    )

    # Load CSV agents
    swarm.load_from_csv()

    # Run with same task for all agents
    result = swarm.run("Analyze the impact of AI on education")

    print(f"Tasks completed: {result['tasks_completed']}")
    print(f"Number of agents: {result['number_of_agents']}")

    # Clean up
    import os

    if os.path.exists("research_agents.csv"):
        os.remove("research_agents.csv")

    return result


if __name__ == "__main__":
    # Run all examples
    result1 = example_with_agents()
    result2 = example_with_csv()
    result3 = example_mixed_usage()

    print("\n=== Summary ===")
    print(
        f"Example 1 - Pre-configured agents: {result1['tasks_completed']} tasks"
    )
    print(
        f"Example 2 - CSV configuration: {result2['tasks_completed']} tasks"
    )
    print(
        f"Example 3 - Mixed usage: {result3['tasks_completed']} tasks"
    )
