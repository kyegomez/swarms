import time
from swarms.structs.graph_workflow import GraphWorkflow
from swarms.structs.agent import Agent

NUM_AGENTS = 30

agents = [
    Agent(
        agent_name=f"Agent-{i:02d}",
        agent_description=f"Agent number {i} in large-scale workflow",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
    )
    for i in range(NUM_AGENTS)
]

workflow = GraphWorkflow(
    name="Large-Scale-Workflow",
    description=f"Large-scale workflow with {NUM_AGENTS} agents using rustworkx",
    backend="rustworkx",
    verbose=False,
)

start_time = time.time()
for agent in agents:
    workflow.add_node(agent)
add_nodes_time = time.time() - start_time

start_time = time.time()
for i in range(9):
    workflow.add_edge(agents[i], agents[i + 1])

workflow.add_edges_from_source(
    agents[5],
    agents[10:20],
)

workflow.add_edges_to_target(
    agents[10:20],
    agents[20],
)

for i in range(20, 29):
    workflow.add_edge(agents[i], agents[i + 1])

add_edges_time = time.time() - start_time

start_time = time.time()
workflow.compile()
compile_time = time.time() - start_time

print(
    f"Agents: {len(workflow.nodes)}, Edges: {len(workflow.edges)}, Layers: {len(workflow._sorted_layers)}"
)
print(
    f"Node addition: {add_nodes_time:.4f}s, Edge addition: {add_edges_time:.4f}s, Compilation: {compile_time:.4f}s"
)
print(
    f"Total setup: {add_nodes_time + add_edges_time + compile_time:.4f}s"
)
