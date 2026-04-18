"""
graph_workflow_serialization.py

Demonstrates the new serialization / deserialization API added to GraphWorkflow:

  workflow.to_spec()           -> lightweight topology dict (no agent objects)
  workflow.save_spec(path)     -> write that dict to a JSON file
  GraphWorkflow.from_topology_spec(spec, registry) -> rebuild from dict + agents

Round-trip:
  1. Build a workflow programmatically.
  2. Save its topology to "workflow_spec.json".
  3. Reconstruct an identical workflow from that file + the same agent objects.
  4. Run both workflows on the same task and compare outputs.
"""

import json
import os

from swarms.structs.agent import Agent
from swarms.structs.graph_workflow import GraphWorkflow

# ---------------------------------------------------------------------------
# 1. Build agents
# ---------------------------------------------------------------------------
researcher = Agent(
    agent_name="Researcher",
    model_name="claude-sonnet-4-5",
    agent_description="Gathers and summarises relevant information on the topic.",
    max_loops=1,
    temperature=1,
)

analyst = Agent(
    agent_name="Analyst",
    model_name="claude-sonnet-4-5",
    agent_description="Analyses the researcher's findings and draws insights.",
    max_loops=1,
    temperature=1,
)

writer = Agent(
    agent_name="Writer",
    model_name="claude-sonnet-4-5",
    agent_description="Turns the analyst's insights into a clear, concise report.",
    max_loops=1,
    temperature=1,
)

# ---------------------------------------------------------------------------
# 2. Build the original workflow: Researcher -> Analyst -> Writer
# ---------------------------------------------------------------------------
original = GraphWorkflow(
    name="Research Pipeline",
    description="A three-stage research, analysis, and writing pipeline.",
    max_loops=1,
    verbose=False,
)

original.add_nodes([researcher, analyst, writer])
original.add_edge("Researcher", "Analyst")
original.add_edge("Analyst", "Writer")
original.compile()

# ---------------------------------------------------------------------------
# 3. Serialize the topology to a JSON file (no agent objects needed)
# ---------------------------------------------------------------------------
spec_path = os.path.join(
    os.path.dirname(__file__), "workflow_spec.json"
)
original.save_spec(spec_path)
print(f"Spec saved to: {spec_path}")

# Inspect what was saved
with open(spec_path) as f:
    saved = json.load(f)
print("\n--- Saved spec ---")
print(json.dumps(saved, indent=2))

# ---------------------------------------------------------------------------
# 4. Reconstruct the workflow from the spec file + an agent registry
# ---------------------------------------------------------------------------
# The registry maps agent_name strings to live Agent objects.
# In a real scenario these could be freshly constructed from config.
agent_registry = {
    "Researcher": researcher,
    "Analyst": analyst,
    "Writer": writer,
}

with open(spec_path) as f:
    spec = json.load(f)

reconstructed = GraphWorkflow.from_topology_spec(
    spec,
    agent_registry,
    verbose=False,
)
reconstructed.compile()

print("\n--- Reconstructed workflow nodes ---")
for node_id in reconstructed.nodes:
    print(f"  {node_id}")

print("\n--- Reconstructed workflow edges ---")
for edge in reconstructed.edges:
    print(f"  {edge.source} -> {edge.target}")

# ---------------------------------------------------------------------------
# 5. Run both workflows and show results
# ---------------------------------------------------------------------------
TASK = "Summarise the main benefits of multi-agent AI systems in three bullet points."

print("\n--- Running original workflow ---")
original_result = original.run(TASK)
print(original_result)

print("\n--- Running reconstructed workflow ---")
reconstructed_result = reconstructed.run(TASK)
print(reconstructed_result)
