from swarms.structs.graph_workflow import GraphWorkflow
from swarms.structs.agent import Agent

market_researcher = Agent(
    agent_name="Market-Researcher",
    agent_description="Conducts comprehensive market research and data collection",
    model_name="gpt-4o-mini",
    max_loops=1,
    verbose=False,
)

competitor_analyst = Agent(
    agent_name="Competitor-Analyst",
    agent_description="Analyzes competitor landscape and positioning",
    model_name="gpt-4o-mini",
    max_loops=1,
    verbose=False,
)

market_analyst = Agent(
    agent_name="Market-Analyst",
    agent_description="Analyzes market trends and opportunities",
    model_name="gpt-4o-mini",
    max_loops=1,
    verbose=False,
)

financial_analyst = Agent(
    agent_name="Financial-Analyst",
    agent_description="Analyzes financial metrics and projections",
    model_name="gpt-4o-mini",
    max_loops=1,
    verbose=False,
)

risk_analyst = Agent(
    agent_name="Risk-Analyst",
    agent_description="Assesses market risks and challenges",
    model_name="gpt-4o-mini",
    max_loops=1,
    verbose=False,
)

strategy_consultant = Agent(
    agent_name="Strategy-Consultant",
    agent_description="Develops strategic recommendations based on all analyses",
    model_name="gpt-4o-mini",
    max_loops=1,
    verbose=False,
)

report_writer = Agent(
    agent_name="Report-Writer",
    agent_description="Compiles comprehensive market research report",
    model_name="gpt-4o-mini",
    max_loops=1,
    verbose=False,
)

executive_summary_writer = Agent(
    agent_name="Executive-Summary-Writer",
    agent_description="Creates executive summary for leadership",
    model_name="gpt-4o-mini",
    max_loops=1,
    verbose=False,
)

workflow = GraphWorkflow(
    name="Market-Research-Workflow",
    description="Real-world market research workflow using rustworkx backend",
    backend="rustworkx",
    verbose=False,
)

all_agents = [
    market_researcher,
    competitor_analyst,
    market_analyst,
    financial_analyst,
    risk_analyst,
    strategy_consultant,
    report_writer,
    executive_summary_writer,
]

for agent in all_agents:
    workflow.add_node(agent)

workflow.add_parallel_chain(
    [market_researcher, competitor_analyst],
    [market_analyst, financial_analyst, risk_analyst],
)

workflow.add_edges_to_target(
    [market_analyst, financial_analyst, risk_analyst],
    strategy_consultant,
)

workflow.add_edges_from_source(
    strategy_consultant,
    [report_writer, executive_summary_writer],
)

workflow.add_edges_to_target(
    [market_analyst, financial_analyst, risk_analyst],
    report_writer,
)

task = """
Conduct a comprehensive market research analysis on the electric vehicle (EV) industry:
1. Research current market size, growth trends, and key players
2. Analyze competitor landscape and market positioning
3. Assess financial opportunities and investment potential
4. Evaluate risks and challenges in the EV market
5. Develop strategic recommendations
6. Create detailed report and executive summary
"""

results = workflow.run(task=task)

for agent_name, output in results.items():
    print(f"{agent_name}: {output}")
