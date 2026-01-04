"""
Swarm Rearrangement Example

This example demonstrates SwarmRearrange for rearranging swarms
of swarms with dynamic flow patterns.
"""

from swarms import (
    Agent,
    RoundRobinSwarm,
    SequentialWorkflow,
    SwarmRearrange,
)

agent1 = Agent(
    agent_name="Agent1",
    model_name="gpt-4o-mini",
    max_loops=1,
)

agent2 = Agent(
    agent_name="Agent2",
    model_name="gpt-4o-mini",
    max_loops=1,
)

swarm1 = RoundRobinSwarm(
    name="Swarm1",
    agents=[agent1, agent2],
    max_loops=1,
)

agent3 = Agent(
    agent_name="Agent3",
    model_name="gpt-4o-mini",
    max_loops=1,
)

agent4 = Agent(
    agent_name="Agent4",
    model_name="gpt-4o-mini",
    max_loops=1,
)

swarm2 = SequentialWorkflow(
    name="Swarm2",
    agents=[agent3, agent4],
    max_loops=1,
)

flow = "Swarm1 -> Swarm2"

swarm_rearrange = SwarmRearrange(
    swarms=[swarm1, swarm2],
    flow=flow,
    max_loops=1,
)

task = "Process this task through multiple swarms"
result = swarm_rearrange.run(task)

print("Swarm Rearrangement Result:")
print(result)
