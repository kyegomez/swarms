"""
Graph Workflow with Rustworkx Backend Example

This example demonstrates GraphWorkflow using the rustworkx backend
for 5-10x faster performance on large-scale workflows. Shows batch
agent addition and parallel execution patterns.
"""

from swarms import Agent, GraphWorkflow

research_agent = Agent(
    agent_name="ResearchAgent",
    model_name="gpt-4o-mini",
    system_prompt="You are a research specialist. Gather and analyze information.",
    max_loops=1,
)

analysis_agent = Agent(
    agent_name="AnalysisAgent",
    model_name="gpt-4o-mini",
    system_prompt="You are an analyst. Process research findings and extract insights.",
    max_loops=1,
)

synthesis_agent = Agent(
    agent_name="SynthesisAgent",
    model_name="gpt-4o-mini",
    system_prompt="You synthesize information into comprehensive reports.",
    max_loops=1,
)

workflow = GraphWorkflow(
    name="Research-Analysis-Pipeline",
    backend="rustworkx",
    verbose=True,
)

workflow.add_nodes([research_agent, analysis_agent, synthesis_agent])
workflow.add_edge("ResearchAgent", "AnalysisAgent")
workflow.add_edge("AnalysisAgent", "SynthesisAgent")

task = "What are the latest trends in renewable energy technology?"
results = workflow.run(task)

print("Graph Workflow Results:")
print(results)
