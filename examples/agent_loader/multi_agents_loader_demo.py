from swarms.utils import load_agents_from_markdown

agents = load_agents_from_markdown([
    "market_researcher.md",
    "financial_analyst.md", 
    "risk_analyst.md"
])

# Example 3: Use agents in a workflow
from swarms.structs.sequential_workflow import SequentialWorkflow

workflow = SequentialWorkflow(
    agents=agents,
    max_loops=1
)

task = """
Analyze the AI healthcare market for a $50M investment opportunity.
Focus on market size, competition, financials, and risks.
"""

result = workflow.run(task)