import sys
from swarms import Agent, GraphWorkflow, Edge, Node, NodeType

# --- Agents ---
analyst = Agent(
    agent_name="Analyst",
    agent_description="Analyses raw data and extracts key statistics",
    model_name="gpt-5.4-mini",
    max_loops=1,
    persistent_memory=False,
)

writer = Agent(
    agent_name="Writer",
    agent_description="Turns analysis into a concise executive summary",
    model_name="gpt-5.4-mini",
    max_loops=1,
    persistent_memory=False,
)

reviewer = Agent(
    agent_name="Reviewer",
    agent_description="Checks the summary for accuracy and clarity",
    model_name="gpt-5.4-mini",
    max_loops=1,
    persistent_memory=False,
)

# --- Build the graph ---
workflow = GraphWorkflow()

workflow.add_node(
    Node(id="analyst", type=NodeType.AGENT, agent=analyst)
)
workflow.add_node(
    Node(id="writer", type=NodeType.AGENT, agent=writer)
)
workflow.add_node(
    Node(id="reviewer", type=NodeType.AGENT, agent=reviewer)
)

workflow.add_edge(Edge(source="analyst", target="writer"))
workflow.add_edge(Edge(source="writer", target="reviewer"))

workflow.set_entry_points(["analyst"])
workflow.set_end_points(["reviewer"])

# --- Callbacks ---
completed_nodes: list[str] = []


def on_node_complete(node_name: str, result: str) -> None:
    completed_nodes.append(node_name)
    print(
        f"\n[on_node_complete] '{node_name}' finished ({len(result)} chars)"
    )


def streaming_callback(token: str) -> None:
    sys.stdout.write(token)
    sys.stdout.flush()


# --- Run ---
print("=== GraphWorkflow with callbacks ===\n")
results = workflow.run(
    task=(
        "Dataset: Q1 2026 revenue $4.2 M (+18 % YoY), "
        "churn 3.1 %, NPS 62. "
        "Analyse, summarise, and review."
    ),
    on_node_complete=on_node_complete,
    streaming_callback=streaming_callback,
)

print(f"\n\n=== Completed nodes in order: {completed_nodes} ===")
print("\n=== Final outputs ===")
for node_name, output in results.items():
    print(f"\n[{node_name}]\n{output[:300]}...")
