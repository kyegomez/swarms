"""
Quick Start: Agent Caching with Multiple Agents

This is a simple example showing how to use agent caching with your existing agents
for super fast loading and reuse.
"""

import time
from swarms import Agent
from swarms.utils.agent_cache import cached_agent_loader


def main():
    """Simple example of caching multiple agents."""

    # Create your agents as usual
    agents = [
        Agent(
            agent_name="Quantitative-Trading-Agent",
            agent_description="Advanced quantitative trading and algorithmic analysis agent",
            system_prompt="""You are an expert quantitative trading agent with deep expertise in:
            - Algorithmic trading strategies and implementation
            - Statistical arbitrage and market making
            - Risk management and portfolio optimization
            - High-frequency trading systems
            - Market microstructure analysis
            
            Your core responsibilities include:
            1. Developing and backtesting trading strategies
            2. Analyzing market data and identifying alpha opportunities
            3. Implementing risk management frameworks
            4. Optimizing portfolio allocations
            5. Conducting quantitative research
            6. Monitoring market microstructure
            7. Evaluating trading system performance
            
            You maintain strict adherence to:
            - Mathematical rigor in all analyses
            - Statistical significance in strategy development
            - Risk-adjusted return optimization
            - Market impact minimization
            - Regulatory compliance
            - Transaction cost analysis
            - Performance attribution
            
            You communicate in precise, technical terms while maintaining clarity for stakeholders.""",
            max_loops=1,
            model_name="gpt-4o-mini",
            dynamic_temperature_enabled=True,
            output_type="str-all-except-first",
            streaming_on=True,
            print_on=True,
            telemetry_enable=False,
        ),
        Agent(
            agent_name="Risk-Manager",
            system_prompt="You are a risk management specialist.",
            max_loops=1,
            model_name="gpt-4o-mini",
        ),
        Agent(
            agent_name="Market-Analyst",
            system_prompt="You are a market analysis expert.",
            max_loops=1,
            model_name="gpt-4o-mini",
        ),
    ]

    print(f"Created {len(agents)} agents")

    # BEFORE: Creating agents each time (slow)
    print("\n=== Without Caching (Slow) ===")
    start_time = time.time()
    # Simulate creating new agents each time
    for _ in range(3):
        new_agents = []
        for agent in agents:
            new_agent = Agent(
                agent_name=agent.agent_name,
                system_prompt=agent.system_prompt,
                max_loops=agent.max_loops,
                model_name=agent.model_name,
            )
            new_agents.append(new_agent)
    no_cache_time = time.time() - start_time
    print(f"🐌 Time without caching: {no_cache_time:.3f}s")

    # AFTER: Using cached agents (super fast!)
    print("\n=== With Caching (Super Fast!) ===")

    # First call - will cache the agents
    start_time = time.time()
    cached_agent_loader(agents)
    first_cache_time = time.time() - start_time
    print(f"🔧 First cache load: {first_cache_time:.3f}s")

    # Subsequent calls - retrieves from cache (lightning fast!)
    cache_times = []
    for i in range(3):
        start_time = time.time()
        cached_agents = cached_agent_loader(agents)
        cache_time = time.time() - start_time
        cache_times.append(cache_time)
        print(f"⚡ Cache load #{i+1}: {cache_time:.4f}s")

    avg_cache_time = sum(cache_times) / len(cache_times)

    print("\n📊 Results:")
    print(f"   🐌 Without caching: {no_cache_time:.3f}s")
    print(f"   🚀 With caching:    {avg_cache_time:.4f}s")
    print(
        f"   💨 Speedup:         {no_cache_time/avg_cache_time:.0f}x faster!"
    )

    # Now use your cached agents normally
    print("\n🎯 Using cached agents:")
    task = "What are the best top 3 etfs for gold coverage?"

    for agent in cached_agents[
        :1
    ]:  # Just use the first agent for demo
        print(f"   Running {agent.agent_name}...")
        response = agent.run(task)
        print(f"   Response: {response[:100]}...")


if __name__ == "__main__":
    main()
