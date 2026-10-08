from swarms import Agent, GraphWorkflow

# Build workflow
wf = GraphWorkflow()
wf.add_node(Agent(agent_name="Analyst", model_name="gpt-5.4"))
wf.add_node(
    Agent(agent_name="Writer", model_name="claude-sonnet-4-6")
)
wf.add_edge("Analyst", "Writer")

# Save topology to JSON
wf.save_spec("my_workflow.json")

# Later: reconstruct from JSON (no agents need to be rebuilt manually)
wf2 = GraphWorkflow.from_topology_spec(
    "my_workflow.json", agents=[...]
)
wf2.run("Analyze Q3 earnings and write a summary")
