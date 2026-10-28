"""
Nested Swarms Example

This example demonstrates AgentRearrange running a flow whose
steps are whole swarms, referenced by their names.
"""

from swarms import (
    Agent,
    AgentRearrange,
    RoundRobinSwarm,
    SequentialWorkflow,
)

agent1 = Agent(
    agent_name="Agent1",
    model_name="gpt-5.4",
    max_loops=1,
)

agent2 = Agent(
    agent_name="Agent2",
    model_name="gpt-5.4",
    max_loops=1,
)

swarm1 = RoundRobinSwarm(
    name="Swarm1",
    agents=[agent1, agent2],
    max_loops=1,
)

agent3 = Agent(
    agent_name="Agent3",
    model_name="gpt-5.4",
    max_loops=1,
)

agent4 = Agent(
    agent_name="Agent4",
    model_name="gpt-5.4",
    max_loops=1,
)

swarm2 = SequentialWorkflow(
    name="Swarm2",
    agents=[agent3, agent4],
    max_loops=1,
)

flow = "Swarm1 -> Swarm2"

nested_swarms = AgentRearrange(
    agents=[swarm1, swarm2],
    flow=flow,
    max_loops=1,
)

task = "Process this task through multiple swarms"
result = nested_swarms.run(task)

print("Nested Swarms Result:")
print(result)
