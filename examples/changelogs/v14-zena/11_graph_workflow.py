"""Zena: GraphWorkflow on native rustworkx.

The API is unchanged. Underneath, Python reimplementations of topological
sort and layer computation were replaced with native rustworkx calls,
compile and validation collapsed into a single adjacency pass, one thread
pool is shared per run, and single-node layers execute inline.

This is the fan-out/fan-in shape: research feeds a summarizer and a critic
in parallel, and both feed a final editor.
"""

from swarms import Agent, GraphWorkflow

wf = GraphWorkflow(auto_compile=True)

for name in ["research", "summarize", "critique", "editor"]:
    wf.add_node(
        Agent(agent_name=name, model_name="gpt-5.4-mini", max_loops=1)
    )

wf.add_edge("research", "summarize")
wf.add_edge("research", "critique")
wf.add_edge("summarize", "editor")
wf.add_edge("critique", "editor")

wf.set_entry_points(["research"])
wf.set_end_points(["editor"])

print(wf.run(task="Assess the market for solid-state batteries."))
