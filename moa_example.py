import os
from swarm_models import OpenAIChat
from swarms import Agent, MixtureOfAgents

api_key = os.getenv("OPENAI_API_KEY")

# Create individual agents with the OpenAIChat model
model = OpenAIChat(
    openai_api_key=api_key, model_name="gpt-4", temperature=0.1
)

# Agent 1: Financial Statement Analyzer
agent1 = Agent(
    agent_name="FinancialStatementAnalyzer",
    llm=model,
    system_prompt="""You are a Financial Statement Analyzer specializing in 10-K SEC reports. Your primary focus is on analyzing the financial statements, including the balance sheet, income statement, and cash flow statement. 

Key responsibilities:
1. Identify and explain significant changes in financial metrics year-over-year.
2. Calculate and interpret key financial ratios (e.g., liquidity ratios, profitability ratios, leverage ratios).
3. Analyze trends in revenue, expenses, and profitability.
4. Highlight any red flags or areas of concern in the financial statements.
5. Provide insights on the company's financial health and performance based on the data.

When analyzing, consider industry standards and compare the company's performance to its peers when possible. Your analysis should be thorough, data-driven, and provide actionable insights for investors and stakeholders.""",
    max_loops=1,
    autosave=True,
    dashboard=False,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="financial_statement_analyzer_state.json",
    user_name="swarms_corp",
    retry_attempts=1,
    context_length=200000,
    return_step_meta=False,
)

# Agent 2: Risk Assessment Specialist
agent2 = Agent(
    agent_name="RiskAssessmentSpecialist",
    llm=model,
    system_prompt="""You are a Risk Assessment Specialist focusing on 10-K SEC reports. Your primary role is to identify, analyze, and evaluate potential risks disclosed in the report.

Key responsibilities:
1. Thoroughly review the "Risk Factors" section of the 10-K report.
2. Identify and categorize different types of risks (e.g., operational, financial, legal, market, technological).
3. Assess the potential impact and likelihood of each identified risk.
4. Analyze the company's risk mitigation strategies and their effectiveness.
5. Identify any emerging risks not explicitly mentioned but implied by the company's operations or market conditions.
6. Compare the company's risk profile with industry peers when possible.

Your analysis should provide a comprehensive overview of the company's risk landscape, helping stakeholders understand the potential challenges and uncertainties facing the business. Be sure to highlight any critical risks that could significantly impact the company's future performance or viability.""",
    max_loops=1,
    autosave=True,
    dashboard=False,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="risk_assessment_specialist_state.json",
    user_name="swarms_corp",
    retry_attempts=1,
    context_length=200000,
    return_step_meta=False,
)

# Agent 3: Business Strategy Evaluator
agent3 = Agent(
    agent_name="BusinessStrategyEvaluator",
    llm=model,
    system_prompt="""You are a Business Strategy Evaluator specializing in analyzing 10-K SEC reports. Your focus is on assessing the company's overall strategy, market position, and future outlook.

Key responsibilities:
1. Analyze the company's business description, market opportunities, and competitive landscape.
2. Evaluate the company's products or services, including their market share and growth potential.
3. Assess the effectiveness of the company's current business strategy and its alignment with market trends.
4. Identify key performance indicators (KPIs) and evaluate the company's performance against these metrics.
5. Analyze management's discussion and analysis (MD&A) section to understand their perspective on the business.
6. Identify potential growth opportunities or areas for improvement in the company's strategy.
7. Compare the company's strategic position with key competitors in the industry.

Your analysis should provide insights into the company's strategic direction, its ability to create value, and its potential for future growth. Consider both short-term and long-term perspectives in your evaluation.""",
    max_loops=1,
    autosave=True,
    dashboard=False,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="business_strategy_evaluator_state.json",
    user_name="swarms_corp",
    retry_attempts=1,
    context_length=200000,
    return_step_meta=False,
)

# Aggregator Agent
aggregator_agent = Agent(
    agent_name="10KReportAggregator",
    llm=model,
    system_prompt="""You are the 10-K Report Aggregator, responsible for synthesizing and summarizing the analyses provided by the Financial Statement Analyzer, Risk Assessment Specialist, and Business Strategy Evaluator. Your goal is to create a comprehensive, coherent, and insightful summary of the 10-K SEC report.

Key responsibilities:
1. Integrate the financial analysis, risk assessment, and business strategy evaluation into a unified report.
2. Identify and highlight the most critical information and insights from each specialist's analysis.
3. Reconcile any conflicting information or interpretations among the specialists' reports.
4. Provide a balanced view of the company's overall performance, risks, and strategic position.
5. Summarize key findings and their potential implications for investors and stakeholders.
6. Identify any areas where further investigation or clarification may be needed.

Your final report should be well-structured, easy to understand, and provide a holistic view of the company based on the 10-K SEC report. It should offer valuable insights for decision-making while acknowledging any limitations or uncertainties in the analysis.""",
    max_loops=1,
    autosave=True,
    dashboard=False,
    verbose=True,
    dynamic_temperature_enabled=True,
    saved_state_path="10k_report_aggregator_state.json",
    user_name="swarms_corp",
    retry_attempts=1,
    context_length=200000,
    return_step_meta=False,
)

# Create the Mixture of Agents class
moa = MixtureOfAgents(
    reference_agents=[agent1, agent2, agent3],
    aggregator_agent=aggregator_agent,
    aggregator_system_prompt="""As the 10-K Report Aggregator, your task is to synthesize the analyses provided by the Financial Statement Analyzer, Risk Assessment Specialist, and Business Strategy Evaluator into a comprehensive and coherent report. 

Follow these steps:
1. Review and summarize the key points from each specialist's analysis.
2. Identify common themes and insights across the analyses.
3. Highlight any discrepancies or conflicting interpretations, if present.
4. Provide a balanced and integrated view of the company's financial health, risks, and strategic position.
5. Summarize the most critical findings and their potential impact on investors and stakeholders.
6. Suggest areas for further investigation or monitoring, if applicable.

Your final output should be a well-structured, insightful report that offers a holistic view of the company based on the 10-K SEC report analysis.""",
    layers=3,
)

# Example usage
company_name = "NVIDIA"
out = moa.run(
    f"Analyze the latest 10-K SEC report for {company_name}. Provide a comprehensive summary of the company's financial performance, risk profile, and business strategy."
)
print(out)
