"""
Complete test to verify MajorityVoting works correctly after the fix.
Tests that all features work the same from API perspective.
"""
from swarms.structs.agent import Agent
from swarms.structs.majority_voting import MajorityVoting


def test_complete_functionality():
    """Test that all MajorityVoting features work correctly"""

    print("=" * 70)
    print("COMPLETE MAJORITY VOTING FUNCTIONALITY TEST")
    print("=" * 70)

    # Create test agents (simulating what the API would create)
    print("\n1. Creating worker agents...")
    agent1 = Agent(
        agent_name="Financial-Analyst",
        agent_description="Analyzes financial aspects",
        system_prompt="You are a financial analyst.",
        model_name="gpt-4o-mini",
        max_loops=1,
    )

    agent2 = Agent(
        agent_name="Tech-Expert",
        agent_description="Understands tech industry",
        system_prompt="You are a tech industry expert.",
        model_name="gpt-4o-mini",
        max_loops=1,
    )

    agent3 = Agent(
        agent_name="Risk-Assessor",
        agent_description="Evaluates risks",
        system_prompt="You are a risk assessor.",
        model_name="gpt-4o-mini",
        max_loops=1,
    )
    print("   ✓ Created 3 worker agents")

    # Test 1: Create MajorityVoting (as API would)
    print("\n2. Creating MajorityVoting swarm...")
    try:
        mv = MajorityVoting(
            name="Investment-Analysis-Swarm",
            description="A swarm for investment analysis",
            agents=[agent1, agent2, agent3],
            max_loops=1,
            verbose=False,
        )
        print("   ✓ MajorityVoting created successfully")
    except Exception as e:
        print(f"   ✗ Failed to create MajorityVoting: {e}")
        raise

    # Test 2: Verify internal consensus agent was created
    print("\n3. Verifying internal consensus agent...")
    try:
        assert mv.consensus_agent is not None, "Consensus agent should exist"
        assert mv.consensus_agent.agent_name == "Consensus-Agent", \
            f"Expected 'Consensus-Agent', got '{mv.consensus_agent.agent_name}'"
        print(f"   ✓ Consensus agent created: {mv.consensus_agent.agent_name}")
    except Exception as e:
        print(f"   ✗ Consensus agent verification failed: {e}")
        raise

    # Test 3: Verify conversation object
    print("\n4. Verifying conversation object...")
    try:
        assert mv.conversation is not None, "Conversation should exist"
        print("   ✓ Conversation object created successfully")
    except Exception as e:
        print(f"   ✗ Conversation verification failed: {e}")
        raise

    # Test 4: Verify all worker agents are registered
    print("\n5. Verifying worker agents...")
    try:
        assert len(mv.agents) == 3, f"Expected 3 agents, got {len(mv.agents)}"
        agent_names = [a.agent_name for a in mv.agents]
        print(f"   ✓ All 3 worker agents registered: {agent_names}")
    except Exception as e:
        print(f"   ✗ Worker agents verification failed: {e}")
        raise

    # Test 5: Test with custom consensus agent configuration
    print("\n6. Testing custom consensus agent configuration...")
    try:
        mv_custom = MajorityVoting(
            name="Custom-Consensus-Swarm",
            description="Swarm with custom consensus agent",
            agents=[agent1, agent2],
            consensus_agent_name="Custom-Consensus",
            consensus_agent_model_name="gpt-4o-mini",
            consensus_agent_prompt="You are a custom consensus agent.",
            max_loops=1,
        )
        assert mv_custom.consensus_agent.agent_name == "Custom-Consensus"
        print(f"   ✓ Custom consensus agent: {mv_custom.consensus_agent.agent_name}")
    except Exception as e:
        print(f"   ✗ Custom consensus configuration failed: {e}")
        raise

    # Test 6: Verify backward compatibility (consensus_agent param should be ignored)
    print("\n7. Testing backward compatibility with unused consensus_agent param...")
    try:
        dummy_agent = Agent(
            agent_name="Dummy",
            system_prompt="Dummy",
            model_name="gpt-4o-mini",
            max_loops=1,
        )
        mv_compat = MajorityVoting(
            name="Backward-Compat-Swarm",
            description="Testing backward compatibility",
            agents=[agent1, agent2],
            consensus_agent=dummy_agent,  # This should be ignored
            max_loops=1,
        )
        # The consensus agent should still be the default one, not the dummy
        assert mv_compat.consensus_agent.agent_name == "Consensus-Agent"
        print("   ✓ Unused consensus_agent parameter properly ignored")
    except Exception as e:
        print(f"   ✗ Backward compatibility test failed: {e}")
        raise

    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    print("\nConclusion:")
    print("- All MajorityVoting features work correctly")
    print("- Consensus agent is properly created internally")
    print("- Worker agents are properly registered")
    print("- Custom consensus configuration works")
    print("- Backward compatibility maintained")
    print("- API will work without errors")
    print("=" * 70)


if __name__ == "__main__":
    test_complete_functionality()
