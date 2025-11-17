"""
Multi-Agent Caching Example - Super Fast Agent Loading

This example demonstrates how to use the agent caching system with multiple agents
to achieve 10-100x speedup in agent loading and reuse.
"""

import time
from swarms import Agent
from swarms.utils.agent_cache import (
    cached_agent_loader,
    simple_lru_agent_loader,
    AgentCache,
    get_agent_cache_stats,
    clear_agent_cache,
)


def create_trading_team():
    """Create a team of trading agents."""

    # Create multiple agents for different trading strategies
    agents = [
        Agent(
            agent_name="Quantitative-Trading-Agent",
            agent_description="Advanced quantitative trading and algorithmic analysis agent",
            system_prompt="""You are an expert quantitative trading agent with deep expertise in:
            - Algorithmic trading strategies and implementation
            - Statistical arbitrage and market making
            - Risk management and portfolio optimization
            - High-frequency trading systems
            - Market microstructure analysis""",
            max_loops=1,
            model_name="gpt-4o-mini",
            temperature=0.1,
        ),
        Agent(
            agent_name="Risk-Management-Agent",
            agent_description="Portfolio risk assessment and management specialist",
            system_prompt="""You are a risk management specialist focused on:
            - Portfolio risk assessment and stress testing
            - Value at Risk (VaR) calculations
            - Regulatory compliance monitoring
            - Risk mitigation strategies
            - Capital allocation optimization""",
            max_loops=1,
            model_name="gpt-4o-mini",
            temperature=0.2,
        ),
        Agent(
            agent_name="Market-Analysis-Agent",
            agent_description="Real-time market analysis and trend identification",
            system_prompt="""You are a market analysis expert specializing in:
            - Technical analysis and chart patterns
            - Market sentiment analysis
            - Economic indicator interpretation
            - Trend identification and momentum analysis
            - Support and resistance level identification""",
            max_loops=1,
            model_name="gpt-4o-mini",
            temperature=0.3,
        ),
        Agent(
            agent_name="Options-Trading-Agent",
            agent_description="Options strategies and derivatives trading specialist",
            system_prompt="""You are an options trading specialist with expertise in:
            - Options pricing models and Greeks analysis
            - Volatility trading strategies
            - Complex options spreads and combinations
            - Risk-neutral portfolio construction
            - Derivatives market making""",
            max_loops=1,
            model_name="gpt-4o-mini",
            temperature=0.15,
        ),
        Agent(
            agent_name="ESG-Investment-Agent",
            agent_description="ESG-focused investment analysis and screening",
            system_prompt="""You are an ESG investment specialist focusing on:
            - Environmental, Social, and Governance criteria evaluation
            - Sustainable investment screening
            - Impact investing strategies
            - ESG risk assessment
            - Green finance and climate risk analysis""",
            max_loops=1,
            model_name="gpt-4o-mini",
            temperature=0.25,
        ),
    ]

    return agents


def basic_caching_example():
    """Basic example of caching multiple agents."""
    print("=== Basic Multi-Agent Caching Example ===")

    # Create our trading team
    trading_team = create_trading_team()
    print(f"Created {len(trading_team)} trading agents")

    # First load - agents will be processed and cached
    print("\n🔄 First load (will cache agents)...")
    start_time = time.time()
    cached_team_1 = cached_agent_loader(trading_team)
    first_load_time = time.time() - start_time

    print(
        f"✅ First load: {len(cached_team_1)} agents in {first_load_time:.3f}s"
    )

    # Second load - agents will be retrieved from cache (super fast!)
    print("\n⚡ Second load (from cache)...")
    start_time = time.time()
    cached_team_2 = cached_agent_loader(trading_team)
    second_load_time = time.time() - start_time

    print(
        f"🚀 Second load: {len(cached_team_2)} agents in {second_load_time:.3f}s"
    )
    print(
        f"💨 Speedup: {first_load_time/second_load_time:.1f}x faster!"
    )

    # Show cache statistics
    stats = get_agent_cache_stats()
    print(f"📊 Cache stats: {stats}")

    return cached_team_1


def custom_cache_example():
    """Example using a custom cache for specific use cases."""
    print("\n=== Custom Cache Example ===")

    # Create a custom cache with specific settings
    custom_cache = AgentCache(
        max_memory_cache_size=50,  # Cache up to 50 agents
        cache_dir="trading_team_cache",  # Custom cache directory
        enable_persistent_cache=True,  # Enable disk persistence
        auto_save_interval=120,  # Auto-save every 2 minutes
    )

    # Create agents
    trading_team = create_trading_team()

    # Load with custom cache
    print("🔧 Loading with custom cache...")
    start_time = time.time()
    cached_team = cached_agent_loader(
        trading_team,
        cache_instance=custom_cache,
        parallel_loading=True,
    )
    load_time = time.time() - start_time

    print(f"✅ Loaded {len(cached_team)} agents in {load_time:.3f}s")

    # Get custom cache stats
    stats = custom_cache.get_cache_stats()
    print(f"📊 Custom cache stats: {stats}")

    # Cleanup
    custom_cache.shutdown()

    return cached_team


def simple_lru_example():
    """Example using the simple LRU cache approach."""
    print("\n=== Simple LRU Cache Example ===")

    trading_team = create_trading_team()

    # First load with simple LRU
    print("🔄 First load with simple LRU...")
    start_time = time.time()
    lru_team_1 = simple_lru_agent_loader(trading_team)
    first_time = time.time() - start_time

    # Second load (cached)
    print("⚡ Second load with simple LRU...")
    start_time = time.time()
    simple_lru_agent_loader(trading_team)
    cached_time = time.time() - start_time

    print(
        f"📈 Simple LRU - First: {first_time:.3f}s, Cached: {cached_time:.3f}s"
    )
    print(f"💨 Speedup: {first_time/cached_time:.1f}x faster!")

    return lru_team_1


def team_workflow_simulation():
    """Simulate a real-world workflow with the cached trading team."""
    print("\n=== Team Workflow Simulation ===")

    # Create and cache the team
    trading_team = create_trading_team()
    cached_team = cached_agent_loader(trading_team)

    # Simulate multiple analysis sessions
    tasks = [
        "Analyze the current market conditions for AAPL",
        "What are the top 3 ETFs for gold coverage?",
        "Assess the risk profile of a tech-heavy portfolio",
        "Identify options strategies for volatile markets",
        "Evaluate ESG investment opportunities in renewable energy",
    ]

    print(
        f"🎯 Running {len(tasks)} analysis tasks with {len(cached_team)} agents..."
    )

    session_start = time.time()

    for i, (agent, task) in enumerate(zip(cached_team, tasks)):
        print(f"\n📋 Task {i+1}: {agent.agent_name}")
        print(f"   Question: {task}")

        task_start = time.time()

        # Run the agent on the task
        response = agent.run(task)

        task_time = time.time() - task_start
        print(f"   ⏱️  Completed in {task_time:.2f}s")
        print(
            f"   💡 Response: {response[:100]}..."
            if len(response) > 100
            else f"   💡 Response: {response}"
        )

    total_session_time = time.time() - session_start
    print(f"\n🏁 Total session time: {total_session_time:.2f}s")
    print(
        f"📊 Average task time: {total_session_time/len(tasks):.2f}s"
    )


def performance_comparison():
    """Compare performance with and without caching."""
    print("\n=== Performance Comparison ===")

    # Create test agents
    test_agents = []
    for i in range(10):
        agent = Agent(
            agent_name=f"Test-Agent-{i:02d}",
            model_name="gpt-4o-mini",
            system_prompt=f"You are test agent number {i}.",
            max_loops=1,
        )
        test_agents.append(agent)

    # Test without caching (creating new agents each time)
    print("🔄 Testing without caching...")
    no_cache_times = []
    for _ in range(3):
        start_time = time.time()
        # Simulate creating new agents each time
        new_agents = []
        for agent in test_agents:
            new_agent = Agent(
                agent_name=agent.agent_name,
                model_name=agent.model_name,
                system_prompt=agent.system_prompt,
                max_loops=agent.max_loops,
            )
            new_agents.append(new_agent)
        no_cache_time = time.time() - start_time
        no_cache_times.append(no_cache_time)

    avg_no_cache_time = sum(no_cache_times) / len(no_cache_times)

    # Clear cache for fair comparison
    clear_agent_cache()

    # Test with caching (first load)
    print("🔧 Testing with caching (first load)...")
    start_time = time.time()
    cached_agent_loader(test_agents)
    first_cache_time = time.time() - start_time

    # Test with caching (subsequent loads)
    print("⚡ Testing with caching (subsequent loads)...")
    cache_times = []
    for _ in range(3):
        start_time = time.time()
        cached_agent_loader(test_agents)
        cache_time = time.time() - start_time
        cache_times.append(cache_time)

    avg_cache_time = sum(cache_times) / len(cache_times)

    # Results
    print(f"\n📊 Performance Results for {len(test_agents)} agents:")
    print(f"   🐌 No caching (avg):     {avg_no_cache_time:.4f}s")
    print(f"   🔧 Cached (first load):  {first_cache_time:.4f}s")
    print(f"   🚀 Cached (avg):         {avg_cache_time:.4f}s")
    print(
        f"   💨 Cache speedup:        {avg_no_cache_time/avg_cache_time:.1f}x faster!"
    )

    # Final cache stats
    final_stats = get_agent_cache_stats()
    print(f"   📈 Final cache stats: {final_stats}")


def main():
    """Run all examples to demonstrate multi-agent caching."""
    print("🤖 Multi-Agent Caching System Examples")
    print("=" * 50)

    try:
        # Run examples
        basic_caching_example()
        custom_cache_example()
        simple_lru_example()
        performance_comparison()
        team_workflow_simulation()

        print("\n✅ All examples completed successfully!")
        print("\n🎯 Key Benefits of Multi-Agent Caching:")
        print("• 🚀 10-100x faster agent loading from cache")
        print(
            "• 💾 Persistent disk cache survives application restarts"
        )
        print("• 🧠 Intelligent LRU memory management")
        print("• 🔄 Background preloading for zero-latency access")
        print("• 📊 Detailed performance monitoring")
        print("• 🛡️ Thread-safe with memory leak prevention")
        print("• ⚡ Parallel processing for multiple agents")

    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
