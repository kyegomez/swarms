from swarms.structs.graph_workflow import GraphWorkflow
from swarms.structs.agent import Agent

research_agent = Agent(
    agent_name="Research-Analyst",
    agent_description="Specialized in comprehensive research and data gathering",
    model_name="gpt-4o-mini",
    max_loops=1,
    verbose=False,
)

analysis_agent = Agent(
    agent_name="Data-Analyst",
    agent_description="Expert in data analysis and pattern recognition",
    model_name="gpt-4o-mini",
    max_loops=1,
    verbose=False,
)

strategy_agent = Agent(
    agent_name="Strategy-Consultant",
    agent_description="Specialized in strategic planning and recommendations",
    model_name="gpt-4o-mini",
    max_loops=1,
    verbose=False,
)

workflow = GraphWorkflow(
    name="Rustworkx-Basic-Workflow",
    description="Basic workflow using rustworkx backend for faster graph operations",
    backend="rustworkx",
    verbose=False,
)

workflow.add_node(research_agent)
workflow.add_node(analysis_agent)
workflow.add_node(strategy_agent)

workflow.add_edge(research_agent, analysis_agent)
workflow.add_edge(analysis_agent, strategy_agent)

task = "Conduct a research analysis on water stocks and ETFs"
results = workflow.run(task=task)

for agent_name, output in results.items():
    print(f"{agent_name}: {output}")
