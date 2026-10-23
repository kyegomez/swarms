"""A DAG that posts each node's output to the file as it completes."""

from swarms import (
    Agent,
    Edge,
    FileChannel,
    GraphWorkflow,
    Node,
    NodeType,
)

analyst = Agent(
    agent_name="Analyst", model_name="gpt-5.4", max_loops=1
)
writer = Agent(agent_name="Writer", model_name="gpt-5.4", max_loops=1)
researcher = Agent(
    agent_name="Researcher", model_name="gpt-5.4", max_loops=1
)
editor = Agent(agent_name="Editor", model_name="gpt-5.4", max_loops=1)

channel = FileChannel("runs/graph.md")

wf = GraphWorkflow(channel=channel)
for agent in [analyst, writer, researcher, editor]:
    wf.add_node(
        Node(id=agent.agent_name, type=NodeType.AGENT, agent=agent)
    )

wf.add_edge(Edge(source="Analyst", target="Writer"))
wf.add_edge(Edge(source="Analyst", target="Researcher"))
wf.add_edge(Edge(source="Writer", target="Editor"))
wf.add_edge(Edge(source="Researcher", target="Editor"))
wf.set_entry_points(["Analyst"])
wf.set_end_points(["Editor"])

wf.run(task="Produce a short market report on AI chips.")

print(channel.path.read_text())
