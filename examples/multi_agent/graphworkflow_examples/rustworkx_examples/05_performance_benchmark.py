import time
from swarms.structs.graph_workflow import GraphWorkflow
from swarms.structs.agent import Agent

agents_small = [
    Agent(
        agent_name=f"Agent-{i}",
        agent_description=f"Agent number {i}",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
    )
    for i in range(5)
]

agents_medium = [
    Agent(
        agent_name=f"Agent-{i}",
        agent_description=f"Agent number {i}",
        model_name="gpt-4o-mini",
        max_loops=1,
        verbose=False,
    )
    for i in range(20)
]

nx_workflow_small = GraphWorkflow(
    name="NetworkX-Small",
    backend="networkx",
    verbose=False,
    auto_compile=False,
)

for agent in agents_small:
    nx_workflow_small.add_node(agent)

for i in range(len(agents_small) - 1):
    nx_workflow_small.add_edge(agents_small[i], agents_small[i + 1])

nx_start = time.time()
nx_workflow_small.compile()
nx_small_time = time.time() - nx_start

rx_workflow_small = GraphWorkflow(
    name="Rustworkx-Small",
    backend="rustworkx",
    verbose=False,
    auto_compile=False,
)

for agent in agents_small:
    rx_workflow_small.add_node(agent)

for i in range(len(agents_small) - 1):
    rx_workflow_small.add_edge(agents_small[i], agents_small[i + 1])

rx_start = time.time()
rx_workflow_small.compile()
rx_small_time = time.time() - rx_start

nx_workflow_medium = GraphWorkflow(
    name="NetworkX-Medium",
    backend="networkx",
    verbose=False,
    auto_compile=False,
)

for agent in agents_medium:
    nx_workflow_medium.add_node(agent)

for i in range(len(agents_medium) - 1):
    nx_workflow_medium.add_edge(
        agents_medium[i], agents_medium[i + 1]
    )

nx_start = time.time()
nx_workflow_medium.compile()
nx_medium_time = time.time() - nx_start

rx_workflow_medium = GraphWorkflow(
    name="Rustworkx-Medium",
    backend="rustworkx",
    verbose=False,
    auto_compile=False,
)

for agent in agents_medium:
    rx_workflow_medium.add_node(agent)

for i in range(len(agents_medium) - 1):
    rx_workflow_medium.add_edge(
        agents_medium[i], agents_medium[i + 1]
    )

rx_start = time.time()
rx_workflow_medium.compile()
rx_medium_time = time.time() - rx_start

print(
    f"Small (5 agents) - NetworkX: {nx_small_time:.4f}s, Rustworkx: {rx_small_time:.4f}s, Speedup: {nx_small_time/rx_small_time if rx_small_time > 0 else 0:.2f}x"
)
print(
    f"Medium (20 agents) - NetworkX: {nx_medium_time:.4f}s, Rustworkx: {rx_medium_time:.4f}s, Speedup: {nx_medium_time/rx_medium_time if rx_medium_time > 0 else 0:.2f}x"
)
