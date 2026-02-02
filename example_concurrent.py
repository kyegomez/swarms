"""
Concurrent Multi-Agent Example

This example demonstrates concurrent execution where multiple agents work simultaneously
on different aspects of the same problem, then results are aggregated.

Use Case: Market analysis where multiple agents analyze different sectors in parallel
"""

import asyncio
from swarms import Agent

# Create specialized agents for concurrent analysis
tech_analyst = Agent(
    agent_name="Tech-Analyst",
    agent_description="Expert in technology sector analysis",
    model_name="gpt-4o-mini",
    max_loops=1,
    system_prompt="You are a technology sector analyst. Provide detailed analysis of tech market trends.",
)

finance_analyst = Agent(
    agent_name="Finance-Analyst",
    agent_description="Expert in financial sector analysis",
    model_name="gpt-4o-mini",
    max_loops=1,
    system_prompt="You are a financial sector analyst. Provide detailed analysis of finance market trends.",
)

healthcare_analyst = Agent(
    agent_name="Healthcare-Analyst",
    agent_description="Expert in healthcare sector analysis",
    model_name="gpt-4o-mini",
    max_loops=1,
    system_prompt="You are a healthcare sector analyst. Provide detailed analysis of healthcare market trends.",
)

# Aggregator agent to combine results
aggregator_agent = Agent(
    agent_name="Aggregator-Agent",
    agent_description="Expert at synthesizing multiple analyses into comprehensive reports",
    model_name="gpt-4o-mini",
    max_loops=1,
    system_prompt="You are a senior analyst. Synthesize multiple sector analyses into a unified market report.",
)


async def run_agent_async(agent: Agent, task: str) -> dict:
    """Run an agent asynchronously and return results."""
    result = await asyncio.to_thread(agent.run, task)
    return {
        "agent": agent.agent_name,
        "result": result
    }


async def concurrent_analysis(query: str):
    """Run multiple agents concurrently."""
    print("="*80)
    print("CONCURRENT AGENT EXECUTION")
    print("="*80)
    print(f"\nQuery: {query}\n")
    
    # Create tasks for concurrent execution
    tasks = [
        run_agent_async(tech_analyst, f"Analyze the technology sector: {query}"),
        run_agent_async(finance_analyst, f"Analyze the financial sector: {query}"),
        run_agent_async(healthcare_analyst, f"Analyze the healthcare sector: {query}"),
    ]
    
    print("Running 3 analysts concurrently...")
    print("-"*80)
    
    # Execute all agents concurrently
    results = await asyncio.gather(*tasks)
    
    print("\nAll analyses completed!")
    print("-"*80)
    
    # Display individual results
    for result in results:
        print(f"\n[{result['agent']}] Analysis:")
        print(f"{result['result'][:200]}..." if len(result['result']) > 200 else result['result'])
        print()
    
    # Aggregate results
    print("="*80)
    print("AGGREGATING RESULTS")
    print("="*80)
    
    combined_analysis = "\n\n".join([
        f"{r['agent']} Analysis:\n{r['result']}" 
        for r in results
    ])
    
    final_report = aggregator_agent.run(
        f"Create a comprehensive market report by synthesizing these sector analyses:\n\n{combined_analysis}"
    )
    
    return final_report


if __name__ == "__main__":
    query = "What are the growth opportunities in Q1 2024?"
    
    # Run concurrent analysis
    final_report = asyncio.run(concurrent_analysis(query))
    
    # Display final aggregated report
    print("\n" + "="*80)
    print("FINAL AGGREGATED REPORT")
    print("="*80)
    print(final_report)
    print("\n" + "="*80)
    print("Concurrent workflow completed!")
    print("="*80)
