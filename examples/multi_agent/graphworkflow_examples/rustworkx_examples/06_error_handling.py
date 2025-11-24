from swarms.structs.graph_workflow import GraphWorkflow
from swarms.structs.agent import Agent

test_agent = Agent(
    agent_name="Test-Agent",
    agent_description="Test agent for error handling",
    model_name="gpt-4o-mini",
    max_loops=1,
    verbose=False,
)

workflow_rx = GraphWorkflow(
    name="Rustworkx-Workflow",
    backend="rustworkx",
    verbose=False,
)
workflow_rx.add_node(test_agent)

workflow_nx = GraphWorkflow(
    name="NetworkX-Workflow",
    backend="networkx",
    verbose=False,
)
workflow_nx.add_node(test_agent)

workflow_default = GraphWorkflow(
    name="Default-Workflow",
    verbose=False,
)
workflow_default.add_node(test_agent)

workflow_invalid = GraphWorkflow(
    name="Invalid-Workflow",
    backend="invalid_backend",
    verbose=False,
)
workflow_invalid.add_node(test_agent)

print(
    f"Rustworkx backend: {type(workflow_rx.graph_backend).__name__}"
)
print(f"NetworkX backend: {type(workflow_nx.graph_backend).__name__}")
print(
    f"Default backend: {type(workflow_default.graph_backend).__name__}"
)
print(
    f"Invalid backend fallback: {type(workflow_invalid.graph_backend).__name__}"
)

try:
    import rustworkx as rx

    print("Rustworkx available: True")
except ImportError:
    print("Rustworkx available: False")
