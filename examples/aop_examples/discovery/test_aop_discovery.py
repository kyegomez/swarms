#!/usr/bin/env python3
"""
Test script to verify the agent discovery functionality works correctly.
"""

import sys
import os

# Add the swarms directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "swarms"))

from swarms import Agent
from swarms.structs.aop import AOP


def test_agent_discovery():
    """Test the agent discovery functionality."""

    print("🧪 Testing AOP Agent Discovery Functionality")
    print("=" * 50)

    # Create test agents
    agent1 = Agent(
        agent_name="TestAgent1",
        agent_description="First test agent for discovery",
        system_prompt="This is a test agent with a very long system prompt that should be truncated to 200 characters when returned by the discovery tool. This prompt contains detailed instructions about how the agent should behave and what tasks it can perform.",
        tags=["test", "agent", "discovery"],
        capabilities=["testing", "validation"],
        role="tester",
    )

    agent2 = Agent(
        agent_name="TestAgent2",
        agent_description="Second test agent for discovery",
        system_prompt="Another test agent with different capabilities and a shorter prompt.",
        tags=["test", "agent", "analysis"],
        capabilities=["analysis", "reporting"],
        role="analyst",
    )

    # Create AOP cluster
    aop = AOP(
        server_name="Test Cluster",
        description="Test cluster for agent discovery",
        verbose=False,
    )

    # Add agents
    aop.add_agent(agent1, tool_name="test_agent_1")
    aop.add_agent(agent2, tool_name="test_agent_2")

    print(f"✅ Created AOP cluster with {len(aop.agents)} agents")
    print(f"📋 Available tools: {aop.list_agents()}")
    print()

    # Test discovery functionality
    print("🔍 Testing agent discovery...")

    # Test getting info for specific agent
    agent1_info = aop._get_agent_discovery_info("test_agent_1")
    assert (
        agent1_info is not None
    ), "Should be able to get info for test_agent_1"
    assert (
        agent1_info["agent_name"] == "TestAgent1"
    ), "Agent name should match"
    assert (
        agent1_info["description"] == "First test agent for discovery"
    ), "Description should match"
    assert (
        len(agent1_info["short_system_prompt"]) <= 203
    ), "System prompt should be truncated to ~200 chars"
    assert "test" in agent1_info["tags"], "Tags should include 'test'"
    assert (
        "testing" in agent1_info["capabilities"]
    ), "Capabilities should include 'testing'"
    assert agent1_info["role"] == "tester", "Role should be 'tester'"

    print("✅ Specific agent discovery test passed")

    # Test getting info for non-existent agent
    non_existent_info = aop._get_agent_discovery_info(
        "non_existent_agent"
    )
    assert (
        non_existent_info is None
    ), "Should return None for non-existent agent"

    print("✅ Non-existent agent test passed")

    # Test that discovery tool is registered
    # Note: In a real scenario, this would be tested via MCP tool calls
    # For now, we just verify the method exists and works
    try:
        # This simulates what the MCP tool would do
        discovery_result = {"success": True, "agents": []}

        for tool_name in aop.agents.keys():
            agent_info = aop._get_agent_discovery_info(tool_name)
            if agent_info:
                discovery_result["agents"].append(agent_info)

        assert (
            len(discovery_result["agents"]) == 2
        ), "Should discover both agents"
        assert (
            discovery_result["success"] is True
        ), "Discovery should be successful"

        print("✅ Discovery tool simulation test passed")

    except Exception as e:
        print(f"❌ Discovery tool test failed: {e}")
        return False

    # Test system prompt truncation
    long_prompt = "A" * 300  # 300 character string
    agent_with_long_prompt = Agent(
        agent_name="LongPromptAgent",
        agent_description="Agent with very long system prompt",
        system_prompt=long_prompt,
    )

    aop.add_agent(
        agent_with_long_prompt, tool_name="long_prompt_agent"
    )
    long_prompt_info = aop._get_agent_discovery_info(
        "long_prompt_agent"
    )

    assert (
        long_prompt_info is not None
    ), "Should get info for long prompt agent"
    assert (
        len(long_prompt_info["short_system_prompt"]) == 203
    ), "Should truncate to 200 chars + '...'"
    assert long_prompt_info["short_system_prompt"].endswith(
        "..."
    ), "Should end with '...'"

    print("✅ System prompt truncation test passed")

    # Test with missing attributes
    minimal_agent = Agent(
        agent_name="MinimalAgent",
        # No description, tags, capabilities, or role specified
    )

    aop.add_agent(minimal_agent, tool_name="minimal_agent")
    minimal_info = aop._get_agent_discovery_info("minimal_agent")

    assert (
        minimal_info is not None
    ), "Should get info for minimal agent"
    assert (
        minimal_info["description"] == "No description available"
    ), "Should have default description"
    assert minimal_info["tags"] == [], "Should have empty tags list"
    assert (
        minimal_info["capabilities"] == []
    ), "Should have empty capabilities list"
    assert (
        minimal_info["role"] == "worker"
    ), "Should have default role"

    print("✅ Minimal agent attributes test passed")

    print()
    print(
        "🎉 All tests passed! Agent discovery functionality is working correctly."
    )
    print()
    print("📊 Summary of discovered agents:")
    for tool_name in aop.agents.keys():
        info = aop._get_agent_discovery_info(tool_name)
        if info:
            print(
                f"   • {info['agent_name']} ({info['role']}) - {info['description']}"
            )

    return True


if __name__ == "__main__":
    try:
        success = test_agent_discovery()
        if success:
            print("\n✅ All tests completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Some tests failed!")
            sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
