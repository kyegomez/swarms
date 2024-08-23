from swarms import Agent, Anthropic, AgentRearrange

# Define the agents with specific tasks for financial activities
agent_risk_analysis = Agent(
    agent_name="RiskAnalysis",
    agent_description="Analyze the financial risks associated with the portfolio.",
    system_prompt="Analyze and identify the risks in the financial data provided.",
    llm=Anthropic(),
    max_loops=1,
    autosave=True,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    stopping_token="<DONE>",
)

agent_compliance_check = Agent(
    agent_name="ComplianceCheck",
    agent_description="Ensure all financial activities adhere to regulatory standards.",
    system_prompt="Review the financial data to ensure compliance with all relevant regulations.",
    llm=Anthropic(),
    max_loops=1,
    autosave=True,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    stopping_token="<DONE>",
)

agent_report_generation = Agent(
    agent_name="ReportGeneration",
    agent_description="Generate a detailed report based on the risk analysis and compliance check.",
    system_prompt="Compile the findings from risk analysis and compliance checks into a comprehensive financial report.",
    llm=Anthropic(),
    max_loops=1,
    autosave=True,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    stopping_token="<DONE>",
)

# Initialize the AgentRearrange system
financial_workflow = AgentRearrange(
    agents=[
        agent_risk_analysis,
        agent_compliance_check,
        agent_report_generation,
    ],
    flow="RiskAnalysis -> ComplianceCheck -> ReportGeneration",
    verbose=True,
)

# Run the workflow on a task
default_task = (
    "Prepare a comprehensive financial review for the fiscal quarter."
)
results = financial_workflow.run(default_task)
print("Workflow Results:", results)
