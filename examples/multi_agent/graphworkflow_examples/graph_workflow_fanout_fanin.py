"""
GraphWorkflow fan-out / fan-in.

A research node fans out to a summarizer and a critic that run in parallel, and
both feed a final editor. Fan-out/fan-in is the most common non-linear pattern
in agentic systems, and GraphWorkflow runs the independent branches (summarize
and critique) concurrently by default — no configuration required.

Graph:
    Research ─┬─> Summarize ─┐
              └─> Critique  ─┴─> Final

Agents are passed straight into ``add_node`` / ``add_edge`` (no Node/Edge
wrappers, no stringly-typed names), ``auto_compile=True`` prepares the graph,
and ``run()`` returns a dict keyed by agent name.

Set your model provider key before running, e.g.:
    export OPENAI_API_KEY="your-api-key"

Run:
    uv run examples/multi_agent/graphworkflow_examples/graph_workflow_fanout_fanin.py
"""

from swarms import Agent, GraphWorkflow


def node(name: str, prompt: str) -> Agent:
    return Agent(
        agent_name=name,
        system_prompt=prompt,
        model_name="gpt-4o-mini",
        max_loops=1,
    )


research = node("Research", "Research the given topic thoroughly.")
summarize = node(
    "Summarize", "Summarize the research into key points."
)
critique = node(
    "Critique", "Critique the research and flag weak claims."
)
final = node(
    "Final", "Write the final brief from the summary and critique."
)

wf = GraphWorkflow(name="Analysis", auto_compile=True)

for agent in (research, summarize, critique, final):
    wf.add_node(agent)

wf.add_edge(research, summarize)
wf.add_edge(research, critique)
wf.add_edge(summarize, final)
wf.add_edge(critique, final)

result = wf.run(task="Assess the EV battery market")

# result is a dict keyed by agent name; every intermediate is available too.
print(result["Final"])
