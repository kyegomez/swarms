from swarms.structs.graph_workflow import GraphWorkflow
from swarms.structs.agent import Agent

data_collector_1 = Agent(
    agent_name="Data-Collector-1",
    agent_description="Collects market data",
    model_name="gpt-4o-mini",
    max_loops=1,
    verbose=False,
)

data_collector_2 = Agent(
    agent_name="Data-Collector-2",
    agent_description="Collects financial data",
    model_name="gpt-4o-mini",
    max_loops=1,
    verbose=False,
)

technical_analyst = Agent(
    agent_name="Technical-Analyst",
    agent_description="Performs technical analysis",
    model_name="gpt-4o-mini",
    max_loops=1,
    verbose=False,
)

fundamental_analyst = Agent(
    agent_name="Fundamental-Analyst",
    agent_description="Performs fundamental analysis",
    model_name="gpt-4o-mini",
    max_loops=1,
    verbose=False,
)

risk_analyst = Agent(
    agent_name="Risk-Analyst",
    agent_description="Performs risk analysis",
    model_name="gpt-4o-mini",
    max_loops=1,
    verbose=False,
)

strategy_consultant = Agent(
    agent_name="Strategy-Consultant",
    agent_description="Develops strategic recommendations",
    model_name="gpt-4o-mini",
    max_loops=1,
    verbose=False,
)

report_writer = Agent(
    agent_name="Report-Writer",
    agent_description="Writes comprehensive reports",
    model_name="gpt-4o-mini",
    max_loops=1,
    verbose=False,
)

workflow = GraphWorkflow(
    name="Complex-Multi-Layer-Workflow",
    description="Complex workflow with multiple layers and parallel processing",
    backend="rustworkx",
    verbose=False,
)

all_agents = [
    data_collector_1,
    data_collector_2,
    technical_analyst,
    fundamental_analyst,
    risk_analyst,
    strategy_consultant,
    report_writer,
]

for agent in all_agents:
    workflow.add_node(agent)

workflow.add_parallel_chain(
    [data_collector_1, data_collector_2],
    [technical_analyst, fundamental_analyst, risk_analyst],
)

workflow.add_edges_to_target(
    [technical_analyst, fundamental_analyst, risk_analyst],
    strategy_consultant,
)

workflow.add_edges_to_target(
    [technical_analyst, fundamental_analyst, risk_analyst],
    report_writer,
)

workflow.add_edge(strategy_consultant, report_writer)

task = "Conduct a comprehensive analysis of the renewable energy sector including market trends, financial health, and risk assessment"
results = workflow.run(task=task)

for agent_name, output in results.items():
    print(f"{agent_name}: {output}")
