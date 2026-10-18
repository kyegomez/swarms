from swarms import Agent, GraphWorkflow

if __name__ == "__main__":
    # Initialize two basic agents
    agent1 = Agent(
        agent_name="Agent1",
        model_name="openai/gpt-4o-mini",
        temperature=0.5,
        max_tokens=4000,
        max_loops=1,
        autosave=True,
        dashboard=True,
    )

    agent2 = Agent(
        agent_name="Agent2",
        model_name="openai/gpt-4o-mini",
        temperature=0.5,
        max_tokens=4000,
        max_loops=1,
        autosave=True,
        dashboard=True,
    )

    # Current GraphWorkflow nodes run agents or nested workflows.
    # Use a model-backed summarizer to combine the two entry agents.
    summarizer = Agent(
        agent_name="Summary",
        model_name="openai/gpt-4o-mini",
        system_prompt="Summarize both agents' findings.",
        max_loops=1,
    )

    # Build a fan-in graph using the agents' names as node IDs.
    wf_graph = GraphWorkflow()
    wf_graph.add_nodes([agent1, agent2, summarizer])
    wf_graph.add_edge("Agent1", "Summary")
    wf_graph.add_edge("Agent2", "Summary")

    wf_graph.set_entry_points(["Agent1", "Agent2"])
    wf_graph.set_end_points(["Summary"])

    # Show a text visualization without requiring Graphviz.
    wf_graph.visualize_simple()

    # Execute the graph with a task for the entry agents.
    results = wf_graph.run(
        "Explain the benefits of multi-agent systems."
    )
    print("Execution results:", results)
