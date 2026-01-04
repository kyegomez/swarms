"""
Graph Workflow Batch Agent Addition Example

This example demonstrates GraphWorkflow with batch agent addition
and parallel execution patterns using rustworkx backend.
"""

from swarms import Agent, GraphWorkflow

data_collector_1 = Agent(
    agent_name="Data-Collector-1",
    model_name="gpt-4o-mini",
    max_loops=1,
)

data_collector_2 = Agent(
    agent_name="Data-Collector-2",
    model_name="gpt-4o-mini",
    max_loops=1,
)

data_collector_3 = Agent(
    agent_name="Data-Collector-3",
    model_name="gpt-4o-mini",
    max_loops=1,
)

analyst_1 = Agent(
    agent_name="Analyst-1",
    model_name="gpt-4o-mini",
    max_loops=1,
)

analyst_2 = Agent(
    agent_name="Analyst-2",
    model_name="gpt-4o-mini",
    max_loops=1,
)

analyst_3 = Agent(
    agent_name="Analyst-3",
    model_name="gpt-4o-mini",
    max_loops=1,
)

synthesis = Agent(
    agent_name="Synthesis",
    model_name="gpt-4o-mini",
    max_loops=1,
)

workflow = GraphWorkflow(
    name="Layer-Based-Parallel-Workflow",
    backend="rustworkx",
)

all_agents = [
    data_collector_1,
    data_collector_2,
    data_collector_3,
    analyst_1,
    analyst_2,
    analyst_3,
    synthesis,
]

for agent in all_agents:
    workflow.add_node(agent)

workflow.add_parallel_chain(
    [data_collector_1, data_collector_2, data_collector_3],
    [analyst_1, analyst_2, analyst_3],
)

workflow.add_edges_to_target(
    [analyst_1, analyst_2, analyst_3],
    synthesis,
)

results = workflow.run("Process and analyze data in parallel layers")

print("Graph Workflow Batch Agents Result:")
print(results)
