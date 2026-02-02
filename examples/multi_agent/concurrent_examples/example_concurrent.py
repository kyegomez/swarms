"""
Concurrent Multi-Agent Example

This example demonstrates concurrent execution using ConcurrentWorkflow,
where multiple agents work simultaneously on the same task and results are aggregated.

Use Case: Market analysis where multiple agents analyze different sectors in parallel.
"""

from swarms import Agent
from swarms.structs import ConcurrentWorkflow

# Create specialized agents for concurrent analysis
tech_analyst = Agent(
    agent_name="Tech-Analyst",
    agent_description="Expert in technology sector analysis",
    model_name="gpt-4o-mini",
    max_loops=1,
    system_prompt="You are a technology sector analyst. Provide detailed analysis of tech market trends.",
)

finance_analyst = Agent(
    agent_name="Finance-Analyst",
    agent_description="Expert in financial sector analysis",
    model_name="gpt-4o-mini",
    max_loops=1,
    system_prompt="You are a financial sector analyst. Provide detailed analysis of finance market trends.",
)

healthcare_analyst = Agent(
    agent_name="Healthcare-Analyst",
    agent_description="Expert in healthcare sector analysis",
    model_name="gpt-4o-mini",
    max_loops=1,
    system_prompt="You are a healthcare sector analyst. Provide detailed analysis of healthcare market trends.",
)

# Build the concurrent workflow
workflow = ConcurrentWorkflow(
    agents=[tech_analyst, finance_analyst, healthcare_analyst],
    name="Market-Analysis-Workflow",
    description="Concurrent analysis of technology, finance, and healthcare sectors.",
    output_type="all",
    autosave=True,
    verbose=False,
)

out = workflow.run(
    "Analyze the market for technology, finance, and healthcare sectors"
)
print(out)
