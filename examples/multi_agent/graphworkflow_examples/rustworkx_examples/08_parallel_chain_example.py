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

data_collector_3 = Agent(
    agent_name="Data-Collector-3",
    agent_description="Collects news data",
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

sentiment_analyst = Agent(
    agent_name="Sentiment-Analyst",
    agent_description="Performs sentiment analysis",
    model_name="gpt-4o-mini",
    max_loops=1,
    verbose=False,
)

workflow = GraphWorkflow(
    name="Parallel-Chain-Workflow",
    description="Demonstrates parallel chain pattern with rustworkx",
    backend="rustworkx",
    verbose=False,
)

sources = [data_collector_1, data_collector_2, data_collector_3]
targets = [technical_analyst, fundamental_analyst, sentiment_analyst]

for agent in sources + targets:
    workflow.add_node(agent)

workflow.add_parallel_chain(sources, targets)

workflow.compile()

task = "Analyze the technology sector using multiple data sources and analysis methods"
results = workflow.run(task=task)

for agent_name, output in results.items():
    print(f"{agent_name}: {output}")
