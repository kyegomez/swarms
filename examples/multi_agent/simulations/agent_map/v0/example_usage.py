#!/usr/bin/env python3
"""
Simple example showing how to use the new AgentMapSimulation.run() method.

This demonstrates the task-based simulation feature where you can specify
what topic the agents should discuss when they meet.
"""

from swarms import Agent
from examples.multi_agent.simulations.agent_map.agent_map_simulation import (
    AgentMapSimulation,
)
from examples.multi_agent.simulations.agent_map.v0.demo_simulation import (
    NATURAL_CONVERSATION_PROMPT,
)


def create_simple_agent(name: str, expertise: str) -> Agent:
    """Create a simple agent with natural conversation abilities."""
    system_prompt = f"""You are {name}, an expert in {expertise}. 
    You enjoy meeting and discussing ideas with other professionals.
    {NATURAL_CONVERSATION_PROMPT}"""

    return Agent(
        agent_name=name,
        agent_description=f"Expert in {expertise}",
        system_prompt=system_prompt,
        model_name="gpt-4.1",
        max_loops=1,
        streaming_on=False,
    )


def main():
    """Simple example of task-based agent simulation."""

    print("🚀 Simple Agent Map Simulation Example")
    print("=" * 50)

    # 1. Create the simulation environment
    simulation = AgentMapSimulation(
        map_width=30.0, map_height=30.0, proximity_threshold=6.0
    )

    # 2. Create some agents
    agents = [
        create_simple_agent("Alice", "Machine Learning"),
        create_simple_agent("Bob", "Cybersecurity"),
        create_simple_agent("Carol", "Data Science"),
    ]

    # 3. Add agents to the simulation
    for agent in agents:
        simulation.add_agent(agent, movement_speed=2.0)

    # 4. Define what you want them to discuss
    task = "What are the biggest challenges and opportunities in AI ethics today?"

    # 5. Run the simulation!
    results = simulation.run(
        task=task, duration=180, with_visualization=True  # 3 minutes
    )

    # 6. Check the results
    print("\n📊 Results Summary:")
    print(f"   Conversations: {results['completed_conversations']}")
    print(
        f"   Average length: {results['average_conversation_length']:.1f} exchanges"
    )

    for agent_name, stats in results["agent_statistics"].items():
        print(
            f"   {agent_name}: talked with {len(stats['partners_met'])} other agents"
        )


if __name__ == "__main__":
    main()
