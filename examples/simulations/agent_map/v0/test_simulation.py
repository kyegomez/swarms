#!/usr/bin/env python3
"""
Test script for the Agent Map Simulation.

This script runs a minimal test of the simulation system to validate
that all components work correctly without requiring a GUI.
"""

import time
from swarms import Agent
from simulations.agent_map_simulation import (
    AgentMapSimulation,
    Position,
)


def create_test_agent(name: str) -> Agent:
    """Create a simple test agent."""
    return Agent(
        agent_name=name,
        agent_description=f"Test agent {name}",
        system_prompt=f"""You are {name}, a financial expert. 
        Engage in brief, professional discussions about market topics. 
        Keep responses concise and focused.""",
        model_name="gemini-2.5-flash",
        dynamic_temperature_enabled=True,
        output_type="str-all-except-first",
        streaming_on=False,
        max_loops=1,  # Keep conversations short for testing
        interactive=False,
    )


def test_basic_functionality():
    """Test basic simulation functionality."""
    print("🧪 Testing Agent Map Simulation...")

    # Create simulation
    sim = AgentMapSimulation(
        map_width=20.0,
        map_height=20.0,
        proximity_threshold=5.0,
        update_interval=1.0,
    )

    # Create test agents
    agent1 = create_test_agent("TestTrader")
    agent2 = create_test_agent("TestAnalyst")

    # Add agents to simulation
    sim.add_agent(agent1, Position(10, 10))
    sim.add_agent(
        agent2, Position(12, 10)
    )  # Close enough to trigger conversation

    print("✅ Agents added successfully")

    # Start simulation
    sim.start_simulation()
    print("✅ Simulation started")

    # Let it run briefly
    print("⏳ Running simulation for 10 seconds...")
    time.sleep(10)

    # Check status
    sim.print_status()

    # Stop simulation
    sim.stop_simulation()
    print("✅ Simulation stopped")

    # Check results
    state = sim.get_simulation_state()
    if state["total_conversations"] > 0:
        print(
            f"🎉 Success! {state['total_conversations']} conversations occurred"
        )

        # Save results
        filename = sim.save_conversation_summary("test_results.txt")
        print(f"📄 Test results saved to: {filename}")
    else:
        print("⚠️  No conversations occurred during test")

    return state["total_conversations"] > 0


def test_agent_creation():
    """Test agent creation and configuration."""
    print("\n🧪 Testing agent creation...")

    agent = create_test_agent("TestAgent")

    # Validate agent properties
    assert agent.agent_name == "TestAgent"
    assert agent.model_name == "gemini-2.5-flash"
    assert agent.max_loops == 1

    print("✅ Agent creation test passed")


def test_position_system():
    """Test the position and distance calculation system."""
    print("\n🧪 Testing position system...")

    pos1 = Position(0, 0)
    pos2 = Position(3, 4)

    distance = pos1.distance_to(pos2)
    expected_distance = 5.0  # 3-4-5 triangle

    assert (
        abs(distance - expected_distance) < 0.001
    ), f"Expected {expected_distance}, got {distance}"

    print("✅ Position system test passed")


def test_simulation_state():
    """Test simulation state management."""
    print("\n🧪 Testing simulation state management...")

    sim = AgentMapSimulation(map_width=10, map_height=10)

    # Test empty state
    state = sim.get_simulation_state()
    assert len(state["agents"]) == 0
    assert state["active_conversations"] == 0
    assert state["running"] is False

    # Add agent and test
    agent = create_test_agent("StateTestAgent")
    sim.add_agent(agent)

    state = sim.get_simulation_state()
    assert len(state["agents"]) == 1
    assert "StateTestAgent" in state["agents"]

    # Remove agent and test
    sim.remove_agent("StateTestAgent")
    state = sim.get_simulation_state()
    assert len(state["agents"]) == 0

    print("✅ Simulation state test passed")


def main():
    """Run all tests."""
    print("🚀 Starting Agent Map Simulation Tests")
    print("=" * 50)

    try:
        # Run individual tests
        test_agent_creation()
        test_position_system()
        test_simulation_state()

        # Run full simulation test
        success = test_basic_functionality()

        print("\n" + "=" * 50)
        if success:
            print(
                "🎉 All tests passed! The simulation is working correctly."
            )
        else:
            print("⚠️  Tests completed but no conversations occurred.")
            print(
                "   This might be due to timing or agent positioning."
            )

        print(
            "\n💡 Try running 'python demo_simulation.py --quick' for a more interactive test."
        )

    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    main()
