#!/usr/bin/env python3
"""
Test script for Group Conversation functionality in Agent Map Simulation.

This script demonstrates the new features:
1. Agents can join ongoing conversations
2. Group conversations with multiple participants
3. Enhanced visualization for group discussions
4. Improved status reporting for group interactions

Run this to see agents naturally forming groups and having multi-party conversations!
"""


from swarms import Agent
from examples.multi_agent.simulations.agent_map.agent_map_simulation import (
    AgentMapSimulation,
    Position,
)


def create_test_agent(name: str, specialty: str) -> Agent:
    """
    Create a test agent with specific expertise.

    Args:
        name: Agent name
        specialty: Area of expertise

    Returns:
        Configured test Agent
    """
    system_prompt = f"""You are {name}, an expert in {specialty}.

When meeting colleagues:
- Share insights relevant to your expertise
- Ask thoughtful questions about their work
- Be friendly and collaborative
- Keep responses conversational (1-2 sentences)
- Show interest in group discussions when others join

Focus on building professional relationships and knowledge sharing."""

    return Agent(
        agent_name=name,
        agent_description=f"Expert in {specialty}",
        system_prompt=system_prompt,
        model_name="gpt-4o-mini",
        dynamic_temperature_enabled=True,
        output_type="str",
        streaming_on=False,
        max_loops=1,
    )


def main():
    """Test the group conversation functionality."""
    print("🧪 Testing Group Conversation Features")
    print("=" * 50)

    # Create simulation with settings optimized for group formation
    print("🏗️  Setting up test environment...")
    simulation = AgentMapSimulation(
        map_width=40.0,
        map_height=40.0,
        proximity_threshold=8.0,  # Close enough for conversations
        update_interval=2.0,
    )

    # Adjust group join threshold for easier testing
    simulation.conversation_manager.group_join_threshold = 12.0

    # Create test agents with different specialties
    print("👥 Creating test agents...")
    test_agents = [
        create_test_agent(
            "Alice", "Machine Learning and AI Research"
        ),
        create_test_agent("Bob", "Data Science and Analytics"),
        create_test_agent(
            "Carol", "Software Engineering and Architecture"
        ),
        create_test_agent("David", "Product Management and Strategy"),
        create_test_agent("Eve", "User Experience and Design"),
        create_test_agent("Frank", "DevOps and Infrastructure"),
    ]

    # Add agents to simulation in close proximity to encourage group formation
    positions = [
        Position(20, 20),  # Center cluster
        Position(22, 18),
        Position(18, 22),
        Position(25, 25),  # Secondary cluster
        Position(23, 27),
        Position(27, 23),
    ]

    for i, agent in enumerate(test_agents):
        simulation.add_agent(
            agent=agent,
            position=positions[i],
            movement_speed=1.0,  # Slower movement to encourage conversations
            conversation_radius=8.0,
        )

    print(f"✅ Added {len(test_agents)} test agents")

    # Define a collaborative task
    group_task = """
    COLLABORATIVE DISCUSSION TOPIC:
    
    "Building the Next Generation of AI-Powered Products"
    
    Share your expertise and perspectives on:
    - How your specialty contributes to AI product development
    - Key challenges and opportunities in your domain
    - Best practices for cross-functional collaboration
    - Emerging trends and future directions
    
    Listen to others and build on their ideas!
    """

    try:
        print("\n🚀 Starting group conversation test...")
        print("📋 Topic: Building AI-Powered Products")
        print("🎯 Goal: Test agents joining ongoing conversations")

        # Run the simulation
        results = simulation.run(
            task=group_task,
            duration=120,  # 2 minutes - enough time for group formation
            with_visualization=True,
            update_interval=2.0,
        )

        # Analyze results for group conversation patterns
        print("\n📊 GROUP CONVERSATION TEST RESULTS:")
        print(
            f"🔢 Total Conversations: {results['total_conversations']}"
        )
        print(
            f"✅ Completed Conversations: {results['completed_conversations']}"
        )
        print(
            f"⏱️  Test Duration: {results['duration_seconds']:.1f} seconds"
        )

        # Check for group conversation evidence
        group_conversations = 0
        max_group_size = 0

        for agent_name, stats in results["agent_statistics"].items():
            partners_met = len(stats["partners_met"])
            if partners_met > 1:
                print(
                    f"🤝 {agent_name}: Interacted with {partners_met} different colleagues"
                )

            # Check conversation history for group size indicators
            agent_state = simulation.agents[agent_name]
            for conv in agent_state.conversation_history:
                group_size = conv.get("group_size", 2)
                if group_size > 2:
                    group_conversations += 1
                    max_group_size = max(max_group_size, group_size)

        print("\n🎯 GROUP FORMATION ANALYSIS:")
        if group_conversations > 0:
            print(
                f"✅ SUCCESS: {group_conversations} group conversations detected!"
            )
            print(
                f"👥 Largest group size: {max_group_size} participants"
            )
            print("🎉 Group conversation feature is working!")
        else:
            print(
                "⚠️  No group conversations detected - agents may need more time or closer proximity"
            )
            print(
                "💡 Try running with longer duration or smaller map size"
            )

        print(
            f"\n📄 Detailed conversation log: {results['summary_file']}"
        )

    except Exception as e:
        print(f"\n❌ Test error: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
