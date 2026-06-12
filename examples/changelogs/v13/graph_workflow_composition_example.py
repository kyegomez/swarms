"""GraphWorkflow composition and parallelism (PRs #1620, #1623, #1605).

- Node.from_subgraph embeds a whole GraphWorkflow as a node in another one.
- max_parallel_nodes caps how many nodes execute concurrently.
- validate(raise_on_error=True) fails fast on cycles, orphans, and missing
  entry/end points before anything runs.
"""

from swarms import Agent, GraphWorkflow, Node, Edge

# Inner workflow becomes a single node in the outer one
inner = GraphWorkflow(name="research")
inner.add_node(
    Node.from_agent(
        Agent(
            agent_name="Researcher", model_name="gpt-4.1", max_loops=1
        )
    )
)
inner.set_entry_points(["Researcher"])
inner.set_end_points(["Researcher"])

outer = GraphWorkflow(
    max_parallel_nodes=4
)  # at most 4 nodes run at once
outer.add_node(Node.from_subgraph(inner))  # nested subgraph node
outer.add_node(
    Node.from_agent(
        Agent(agent_name="Writer", model_name="gpt-4.1", max_loops=1)
    )
)
outer.add_edge(Edge(source="research", target="Writer"))
outer.set_entry_points(["research"])
outer.set_end_points(["Writer"])

# Fail fast on cycles, orphans, missing entry points
outer.validate(raise_on_error=True)

result = outer.run(task="Write a brief on AI chips.")
print(result)
