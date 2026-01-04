"""
SwarmRouter with Round Robin Example

This example demonstrates SwarmRouter using RoundRobin routing strategy
for fair, cyclic agent execution with improved communication flow.
"""

from swarms import Agent, SwarmRouter

agent1 = Agent(
    agent_name="Agent1",
    system_prompt="You are a research specialist.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

agent2 = Agent(
    agent_name="Agent2",
    system_prompt="You are an analysis specialist.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

agent3 = Agent(
    agent_name="Agent3",
    system_prompt="You are a synthesis specialist.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

router = SwarmRouter(
    name="Round-Robin-Router",
    agents=[agent1, agent2, agent3],
    swarm_type="RoundRobin",
    max_loops=2,
    verbose=True,
)

task = "Analyze the impact of renewable energy on global markets"
result = router.run(task)

print("SwarmRouter Round Robin Result:")
print(result)
