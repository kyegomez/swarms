#!/usr/bin/env python3
"""
Example showing how agents can use the discovery tool to learn about each other
and collaborate more effectively.
"""

from swarms import Agent
from swarms.structs.aop import AOP


def simulate_agent_discovery():
    """Simulate how an agent would use the discovery tool."""

    # Create the AOP cluster
    aop = AOP(
        server_name="Project Team",
        description="A team of specialized agents for project coordination",
        verbose=True,
    )

    # Add some specialized agents
    data_agent = Agent(
        agent_name="DataSpecialist",
        agent_description="Handles all data-related tasks and analysis",
        system_prompt="You are a data specialist with expertise in data processing, analysis, and visualization. You work with large datasets and create insights.",
        tags=["data", "analysis", "python", "sql", "statistics"],
        capabilities=[
            "data_processing",
            "statistical_analysis",
            "visualization",
        ],
        role="specialist",
    )

    code_agent = Agent(
        agent_name="CodeSpecialist",
        agent_description="Handles all coding and development tasks",
        system_prompt="You are a software development specialist who writes clean, efficient code and follows best practices. You handle both frontend and backend development.",
        tags=[
            "coding",
            "development",
            "python",
            "javascript",
            "react",
        ],
        capabilities=[
            "software_development",
            "code_review",
            "debugging",
        ],
        role="developer",
    )

    writing_agent = Agent(
        agent_name="ContentSpecialist",
        agent_description="Creates and manages all written content",
        system_prompt="You are a content specialist who creates engaging written content, documentation, and marketing materials. You ensure all content is clear and compelling.",
        tags=["writing", "content", "documentation", "marketing"],
        capabilities=[
            "content_creation",
            "technical_writing",
            "editing",
        ],
        role="writer",
    )

    # Add agents to the cluster
    aop.add_agent(data_agent, tool_name="data_specialist")
    aop.add_agent(code_agent, tool_name="code_specialist")
    aop.add_agent(writing_agent, tool_name="content_specialist")

    print("🏢 Project Team AOP Cluster Created!")
    print(f"👥 Team members: {aop.list_agents()}")
    print()

    # Simulate the coordinator discovering team members
    print("🔍 Project Coordinator discovering team capabilities...")
    print()

    # Get discovery info for each agent
    for tool_name in aop.list_agents():
        if (
            tool_name != "discover_agents"
        ):  # Skip the discovery tool itself
            agent_info = aop._get_agent_discovery_info(tool_name)
            if agent_info:
                print(f"📋 {agent_info['agent_name']}:")
                print(f"   Description: {agent_info['description']}")
                print(f"   Role: {agent_info['role']}")
                print(f"   Tags: {', '.join(agent_info['tags'])}")
                print(
                    f"   Capabilities: {', '.join(agent_info['capabilities'])}"
                )
                print(
                    f"   System Prompt: {agent_info['short_system_prompt'][:100]}..."
                )
                print()

    print("💡 How agents would use this in practice:")
    print("   1. Agent calls 'discover_agents' MCP tool")
    print("   2. Gets information about all available agents")
    print(
        "   3. Uses this info to make informed decisions about task delegation"
    )
    print(
        "   4. Can discover specific agents by name for targeted collaboration"
    )
    print()

    # Show what the MCP tool response would look like
    print("📡 Sample MCP tool response structure:")

    print("   discover_agents() -> {")
    print("     'success': True,")
    print("     'agents': [")
    print("       {")
    print("         'tool_name': 'data_specialist',")
    print("         'agent_name': 'DataSpecialist',")
    print(
        "         'description': 'Handles all data-related tasks...',"
    )
    print(
        "         'short_system_prompt': 'You are a data specialist...',"
    )
    print("         'tags': ['data', 'analysis', 'python'],")
    print(
        "         'capabilities': ['data_processing', 'statistics'],"
    )
    print("         'role': 'specialist',")
    print("         ...")
    print("       }")
    print("     ]")
    print("   }")
    print()

    print("✅ Agent discovery system ready for collaborative work!")


if __name__ == "__main__":
    simulate_agent_discovery()
