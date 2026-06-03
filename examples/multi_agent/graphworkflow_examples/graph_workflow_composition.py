"""
GraphWorkflow Composition — nested subgraph example
=====================================================

Demonstrates embedding one GraphWorkflow as a node inside another
(graph-workflow-composition feature).

Topology
--------

    ┌─────────────────────────────────┐
    │  Outer workflow                 │
    │                                 │
    │  [ResearchPipeline subgraph]    │
    │      Researcher ──► Analyst     │
    │             │                   │
    │             ▼                   │
    │       [Report-Writer]           │
    └─────────────────────────────────┘

The inner "ResearchPipeline" graph (Researcher → Analyst) runs as a single
black-box node.  Its merged output is forwarded to the outer Report-Writer
agent as one combined prompt.

Also shows:
- Saving / loading the composed topology with to_spec() / from_topology_spec()
- Automatic checkpoint-dir nesting

Requires ANTHROPIC_API_KEY.
"""

import pprint

from swarms import Agent
from swarms.structs.graph_workflow import GraphWorkflow

# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

researcher = Agent(
    agent_name="Researcher",
    system_prompt="You are a concise research analyst. Respond in 2-3 sentences.",
    model_name="claude-sonnet-4-5",
    max_loops=1,
    print_on=False,
)

analyst = Agent(
    agent_name="Analyst",
    system_prompt=(
        "You receive raw research notes and extract the three most important "
        "data points. Be concise."
    ),
    model_name="claude-sonnet-4-5",
    max_loops=1,
    print_on=False,
)

report_writer = Agent(
    agent_name="Report-Writer",
    system_prompt=(
        "You receive structured analysis and write a one-paragraph executive "
        "summary suitable for a board briefing."
    ),
    model_name="claude-sonnet-4-5",
    max_loops=1,
    print_on=False,
)

# ---------------------------------------------------------------------------
# Inner graph — "ResearchPipeline" subgraph
# ---------------------------------------------------------------------------

research_pipeline = GraphWorkflow(
    name="ResearchPipeline",
    description="Gather research then extract key data points",
)
research_pipeline.add_node(researcher)
research_pipeline.add_node(analyst)
research_pipeline.add_edge("Researcher", "Analyst")
research_pipeline.compile()

# ---------------------------------------------------------------------------
# Outer graph — embeds the subgraph as a single node
# ---------------------------------------------------------------------------

outer = GraphWorkflow(
    name="BoardBriefingWorkflow",
    description="Research pipeline feeds into an executive report writer",
)
outer.add_node(research_pipeline)  # <-- GraphWorkflow as a node
outer.add_node(report_writer)
outer.add_edge("ResearchPipeline", "Report-Writer")
outer.compile()

TASK = "Summarise the current state of large-language-model adoption in enterprise software."

print("=" * 60)
print("Running composed workflow…")
print("=" * 60)

results = outer.run(TASK)

print("\nResults per node:")
pprint.pprint(results)

# ---------------------------------------------------------------------------
# Topology serialisation round-trip
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("Serialising and reconstructing topology…")
print("=" * 60)

spec = outer.to_spec()

# Confirm subgraph is embedded in the spec
subgraph_nodes = [
    n for n in spec["nodes"] if n.get("type") == "subgraph"
]
print(f"Subgraph nodes in spec: {[n['id'] for n in subgraph_nodes]}")
assert len(subgraph_nodes) == 1
assert subgraph_nodes[0]["id"] == "ResearchPipeline"

# Reconstruct — only leaf agents need to be in the registry;
# the subgraph node is self-contained inside the spec.
registry = {
    "Researcher": researcher,
    "Analyst": analyst,
    "Report-Writer": report_writer,
}
rebuilt = GraphWorkflow.from_topology_spec(spec, registry)
assert "ResearchPipeline" in rebuilt.nodes
assert "Report-Writer" in rebuilt.nodes
print("Round-trip reconstruction: OK")
