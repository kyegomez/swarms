"""
GraphWorkflow validate() example.

Demonstrates calling validate() at build time to catch structural
problems before any agent runs.

Scenarios covered:
  1. Clean workflow  — validate() reports no issues.
  2. Isolated node   — an agent with no edges triggers a warning.
  3. Unreachable node — a node not reachable from the entry point triggers a warning.
  4. Invalid agent   — a node with agent=None triggers an error and
                       validate(raise_on_error=True) raises ValueError.
  5. compile() integration — compile() surfaces the same errors as a
                             WARNING-level log so you catch problems
                             without calling validate() explicitly.
"""

from swarms import Agent
from swarms.structs.graph_workflow import GraphWorkflow, Node


def make_agent(name: str, description: str) -> Agent:
    return Agent(
        agent_name=name,
        agent_description=description,
        model_name="gpt-5.4",
        max_loops=1,
        verbose=False,
        print_on=False,
    )


# ---------------------------------------------------------------------------
# Scenario 1 — Clean workflow
# ---------------------------------------------------------------------------
print("\n=== Scenario 1: Clean workflow ===")

researcher = make_agent("Researcher", "Gathers facts on a topic")
analyst = make_agent("Analyst", "Analyses gathered facts")
writer = make_agent("Writer", "Writes a report from analysis")

wf = GraphWorkflow(name="ResearchPipeline")
wf.add_node(researcher)
wf.add_node(analyst)
wf.add_node(writer)
wf.add_edge("Researcher", "Analyst")
wf.add_edge("Analyst", "Writer")

result = wf.validate(raise_on_error=False)
print(f"is_valid : {result['is_valid']}")
print(f"errors   : {result['errors']}")
print(f"warnings : {result['warnings']}")

# ---------------------------------------------------------------------------
# Scenario 2 — Isolated node (no edges)
# ---------------------------------------------------------------------------
print("\n=== Scenario 2: Isolated node ===")

a = make_agent("NodeA", "Does task A")
b = make_agent("NodeB", "Does task B")
orphan = make_agent("Orphan", "Has no edges — will never run")

wf2 = GraphWorkflow(name="IsolatedNodeWorkflow")
wf2.add_node(a)
wf2.add_node(b)
wf2.add_node(orphan)
wf2.add_edge("NodeA", "NodeB")

result2 = wf2.validate(raise_on_error=False)
print(f"is_valid : {result2['is_valid']}")
print(f"warnings : {result2['warnings']}")

# ---------------------------------------------------------------------------
# Scenario 3 — Unreachable node
# ---------------------------------------------------------------------------
print("\n=== Scenario 3: Unreachable node ===")

entry = make_agent("Entry", "Entry point")
mid = make_agent("Mid", "Middle step")
side = make_agent("Side", "Has an edge but unreachable from Entry")

wf3 = GraphWorkflow(name="UnreachableWorkflow")
wf3.add_node(entry)
wf3.add_node(mid)
wf3.add_node(side)
wf3.add_edge("Entry", "Mid")
wf3.add_edge(
    "Side", "Mid"
)  # Side feeds Mid but no path reaches Side from Entry
wf3.set_entry_points(["Entry"])

result3 = wf3.validate(raise_on_error=False)
print(f"is_valid : {result3['is_valid']}")
print(f"warnings : {result3['warnings']}")

# ---------------------------------------------------------------------------
# Scenario 4 — Invalid agent triggers ValueError
# ---------------------------------------------------------------------------
print("\n=== Scenario 4: raise_on_error=True ===")

wf4 = GraphWorkflow(name="InvalidWorkflow")
wf4.nodes["Ghost"] = Node(id="Ghost", agent=None)  # no real agent

try:
    wf4.validate(raise_on_error=True)
except ValueError as exc:
    print(f"Caught ValueError:\n{exc}")

# ---------------------------------------------------------------------------
# Scenario 5 — compile() logs errors automatically
# ---------------------------------------------------------------------------
print("\n=== Scenario 5: compile() integration ===")

wf5 = GraphWorkflow(name="CompileCheckWorkflow")
wf5.nodes["Ghost"] = Node(id="Ghost", agent=None)

# compile() will log a WARNING about the invalid node without raising
wf5.compile()
print("compile() completed — check logs above for validation warning")
