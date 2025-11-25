import time
from swarms.structs.graph_workflow import GraphWorkflow
from swarms.structs.agent import Agent

agents = [
    Agent(
        agent_name=f"Agent-{i}",
        agent_description=f"Agent number {i}",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
    )
    for i in range(5)
]

nx_workflow = GraphWorkflow(
    name="NetworkX-Workflow",
    backend="networkx",
    verbose=False,
)

for agent in agents:
    nx_workflow.add_node(agent)

for i in range(len(agents) - 1):
    nx_workflow.add_edge(agents[i], agents[i + 1])

nx_start = time.time()
nx_workflow.compile()
nx_compile_time = time.time() - nx_start

rx_workflow = GraphWorkflow(
    name="Rustworkx-Workflow",
    backend="rustworkx",
    verbose=False,
)

for agent in agents:
    rx_workflow.add_node(agent)

for i in range(len(agents) - 1):
    rx_workflow.add_edge(agents[i], agents[i + 1])

rx_start = time.time()
rx_workflow.compile()
rx_compile_time = time.time() - rx_start

speedup = (
    nx_compile_time / rx_compile_time if rx_compile_time > 0 else 0
)
print(f"NetworkX compile time: {nx_compile_time:.4f}s")
print(f"Rustworkx compile time: {rx_compile_time:.4f}s")
print(f"Speedup: {speedup:.2f}x")
print(
    f"Identical layers: {nx_workflow._sorted_layers == rx_workflow._sorted_layers}"
)
