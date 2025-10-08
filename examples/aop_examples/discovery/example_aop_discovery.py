#!/usr/bin/env python3
"""
Example demonstrating the new agent discovery MCP tool in AOP.

This example shows how agents can discover information about each other
using the new 'discover_agents' MCP tool.
"""

from swarms import Agent
from swarms.structs.aop import AOP


def main():
    """Demonstrate the agent discovery functionality."""

    # Create some sample agents with different configurations
    agent1 = Agent(
        agent_name="DataAnalyst",
        agent_description="Specialized in data analysis and visualization",
        system_prompt="You are a data analyst with expertise in Python, pandas, and statistical analysis. You help users understand data patterns and create visualizations.",
        tags=["data", "analysis", "python", "pandas"],
        capabilities=["data_analysis", "visualization", "statistics"],
        role="analyst",
        model_name="gpt-4o-mini",
        temperature=0.3,
    )

    agent2 = Agent(
        agent_name="CodeReviewer",
        agent_description="Expert code reviewer and quality assurance specialist",
        system_prompt="You are a senior software engineer who specializes in code review, best practices, and quality assurance. You help identify bugs, suggest improvements, and ensure code follows industry standards.",
        tags=["code", "review", "quality", "python", "javascript"],
        capabilities=[
            "code_review",
            "quality_assurance",
            "best_practices",
        ],
        role="reviewer",
        model_name="gpt-4o-mini",
        temperature=0.2,
    )

    agent3 = Agent(
        agent_name="CreativeWriter",
        agent_description="Creative content writer and storyteller",
        system_prompt="You are a creative writer who specializes in storytelling, content creation, and engaging narratives. You help create compelling stories, articles, and marketing content.",
        tags=["writing", "creative", "content", "storytelling"],
        capabilities=[
            "creative_writing",
            "content_creation",
            "storytelling",
        ],
        role="writer",
        model_name="gpt-4o-mini",
        temperature=0.8,
    )

    # Create AOP cluster with the agents
    aop = AOP(
        server_name="Agent Discovery Demo",
        description="A demo cluster showing agent discovery capabilities",
        agents=[agent1, agent2, agent3],
        verbose=True,
    )

    print("🚀 AOP Cluster initialized with agent discovery tool!")
    print(f"📊 Total agents registered: {len(aop.agents)}")
    print(f"🔧 Available tools: {aop.list_agents()}")
    print()

    # Demonstrate the discovery tool
    print("🔍 Testing agent discovery functionality...")
    print()

    # Test discovering all agents
    print("1. Discovering all agents:")
    all_agents_info = aop._get_agent_discovery_info(
        "DataAnalyst"
    )  # This would normally be called via MCP
    print(
        f"   Found agent: {all_agents_info['agent_name'] if all_agents_info else 'None'}"
    )
    print()

    # Show what the MCP tool would return
    print("2. What the 'discover_agents' MCP tool would return:")
    print("   - Tool name: discover_agents")
    print(
        "   - Description: Discover information about other agents in the cluster"
    )
    print("   - Parameters: agent_name (optional)")
    print(
        "   - Returns: Agent info including name, description, short system prompt, tags, capabilities, role, etc."
    )
    print()

    # Show sample agent info structure
    if all_agents_info:
        print("3. Sample agent discovery info structure:")
        for key, value in all_agents_info.items():
            if key == "short_system_prompt":
                print(f"   {key}: {value[:100]}...")
            else:
                print(f"   {key}: {value}")
        print()

    print("✅ Agent discovery tool successfully integrated!")
    print(
        "💡 Agents can now use the 'discover_agents' MCP tool to learn about each other."
    )
    print(
        "🔄 The tool is automatically updated when new agents are added to the cluster."
    )


if __name__ == "__main__":
    main()
