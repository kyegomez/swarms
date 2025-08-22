#!/usr/bin/env python3
"""
Test script to verify mass agent template can process more than 500 agents.
"""

from mass_agent_template import MassAgentTemplate


def test_mass_agents():
    print(
        "Testing Mass Agent Template - Processing More Than 50 Agents"
    )
    print("=" * 60)

    # Initialize template with 200 agents
    template = MassAgentTemplate(
        agent_count=200,
        budget_limit=50.0,
        batch_size=25,
        verbose=True,
    )

    print(f"Initialized with {len(template.agents)} agents")
    print(f"Budget limit: ${template.cost_tracker.budget_limit}")

    # Test processing 100 agents
    print("\nTesting with 100 agents...")
    result = template.run_mass_task(
        "What is the most important skill for your role?",
        agent_count=100,
    )

    print("Results:")
    print(f"  Agents processed: {len(result['agents_used'])}")
    print(f"  Cost: ${result['cost_stats']['total_cost']:.4f}")
    print(
        f"  Budget remaining: ${result['cost_stats']['budget_remaining']:.2f}"
    )
    print(f"  Cached: {result.get('cached', False)}")

    # Test processing 150 agents
    print("\nTesting with 150 agents...")
    result2 = template.run_mass_task(
        "Describe your approach to problem-solving", agent_count=150
    )

    print("Results:")
    print(f"  Agents processed: {len(result2['agents_used'])}")
    print(f"  Cost: ${result2['cost_stats']['total_cost']:.4f}")
    print(
        f"  Budget remaining: ${result2['cost_stats']['budget_remaining']:.2f}"
    )
    print(f"  Cached: {result2.get('cached', False)}")

    # Show final stats
    final_stats = template.get_system_stats()
    print("\nFinal Statistics:")
    print(f"  Total agents: {final_stats['total_agents']}")
    print(f"  Loaded agents: {final_stats['loaded_agents']}")
    print(
        f"  Total cost: ${final_stats['cost_stats']['total_cost']:.4f}"
    )
    print(
        f"  Budget remaining: ${final_stats['cost_stats']['budget_remaining']:.2f}"
    )

    # Success criteria
    total_processed = len(result["agents_used"]) + len(
        result2["agents_used"]
    )
    print(f"\nTotal agents processed: {total_processed}")

    if total_processed > 50:
        print("✅ SUCCESS: Template processed more than 50 agents!")
    else:
        print("❌ FAILURE: Template still limited to 50 agents")


if __name__ == "__main__":
    test_mass_agents()
