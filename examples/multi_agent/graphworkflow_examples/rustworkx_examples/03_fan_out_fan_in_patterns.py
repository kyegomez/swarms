from swarms import Agent, GraphWorkflow

coordinator = Agent(
    agent_name="Coordinator",
    agent_description="Coordinates and distributes tasks",
    model_name="gpt-4o-mini",
    max_loops=1,
    verbose=False,
)

tech_analyst = Agent(
    agent_name="Tech-Analyst",
    agent_description="Technical analysis specialist",
    model_name="gpt-4o-mini",
    max_loops=1,
    verbose=False,
)

fundamental_analyst = Agent(
    agent_name="Fundamental-Analyst",
    agent_description="Fundamental analysis specialist",
    model_name="gpt-4o-mini",
    max_loops=1,
    verbose=False,
)

sentiment_analyst = Agent(
    agent_name="Sentiment-Analyst",
    agent_description="Sentiment analysis specialist",
    model_name="gpt-4o-mini",
    max_loops=1,
    verbose=False,
)

synthesis_agent = Agent(
    agent_name="Synthesis-Agent",
    agent_description="Synthesizes multiple analyses into final report",
    model_name="gpt-4o-mini",
    max_loops=1,
    verbose=False,
)

workflow = GraphWorkflow(
    name="Fan-Out-Fan-In-Workflow",
    description="Demonstrates parallel processing patterns with rustworkx",
    backend="rustworkx",
    verbose=False,
)

workflow.add_node(coordinator)
workflow.add_node(tech_analyst)
workflow.add_node(fundamental_analyst)
workflow.add_node(sentiment_analyst)
workflow.add_node(synthesis_agent)

workflow.add_edges_from_source(
    coordinator,
    [tech_analyst, fundamental_analyst, sentiment_analyst],
)

workflow.add_edges_to_target(
    [tech_analyst, fundamental_analyst, sentiment_analyst],
    synthesis_agent,
)

task = "Analyze Tesla stock from technical, fundamental, and sentiment perspectives"
results = workflow.run(task=task)

for agent_name, output in results.items():
    print(f"{agent_name}: {output}")


workflow.visualize(view=True)
