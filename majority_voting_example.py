from swarms import Agent, MajorityVoting
from swarms_tools.finance.sector_analysis import macro_sector_analysis

# Initialize multiple agents with a focus on asset allocation and risk management for a $50B portfolio

# print(macro_sector_analysis())

agents = [
    Agent(
        agent_name="Sector-Financial-Analyst",
        agent_description="Senior sector financial analyst focused on optimizing capital allocations for a $50B portfolio at BlackRock.",
        system_prompt="""You are a seasoned financial analyst at BlackRock tasked with optimizing asset allocations from a $50B portfolio. Your responsibilities include:
        - Conducting deep analyses of sector performance, historical trends, and financial fundamentals.
        - Evaluating revenue growth, profitability, and overall market positioning to determine the optimal dollar allocation for each sector.
        - Integrating risk considerations into your fiscal analysis to ensure that recommended capital assignments align with the overall risk tolerance.
        - Presenting a detailed breakdown of how much money should be allocated to each sector and justifying these recommendations with data-driven insights.
        Provide clear, quantitative recommendations in your output, including precise allocation figures for each sector.""",
        max_loops=1,
        model_name="groq/deepseek-r1-distill-qwen-32b",
        max_tokens=3000,
    ),
    Agent(
        agent_name="Sector-Risk-Analyst",
        agent_description="Expert risk management analyst focused on calibrating sector risk allocations for a $50B institutional portfolio.",
        system_prompt="""You are a veteran risk analyst at BlackRock, responsible for defining and advising on risk allocation within a $50B portfolio. Your responsibilities include:
        - Assessing the risk profile and volatility metrics of each market sector.
        - Quantifying risk exposures, performing stress tests, and modeling adverse market scenarios.
        - Recommending precise risk allocation figures (both in absolute and percentage terms) for each sector, ensuring a balanced risk profile across the portfolio.
        - Integrating risk-adjusted return considerations into your analysis so that capital assignments reflect both opportunity and risk mitigation.
        Provide detailed, quantitative insights and clearly articulate how much risk should be assumed per sector relative to the $50B total.""",
        max_loops=1,
        model_name="groq/deepseek-r1-distill-qwen-32b",
        max_tokens=3000,
    ),
    Agent(
        agent_name="Tech-Sector-Analyst",
        agent_description="Specialized analyst focused on the technology sector, tasked with determining capital and risk allocations within the $50B portfolio.",
        system_prompt="""You are a specialized technology sector analyst at BlackRock, focused on the high-growth potential of the tech sector as part of a $50B portfolio. Your responsibilities include:
        - Evaluating current and emerging tech trends, competitive dynamics, and innovation drivers.
        - Analyzing the risk/reward profile of tech investments, including both growth prospects and volatility.
        - Recommending how much capital should be allocated to the technology sector, alongside quantified risk allocations suited to its inherent risk profile.
        - Providing clear, data-backed insights that balance aggressive growth targets with measured risk exposures.
        Deliver a detailed breakdown of your recommendations, including both dollar figures and risk metrics, tailored for the tech sector in the overall portfolio.""",
        max_loops=1,
        model_name="groq/deepseek-r1-distill-qwen-32b",
        max_tokens=3000,
    ),
]

consensus_agent = Agent(
    agent_name="Consensus-Strategist",
    agent_description="Senior strategist who synthesizes allocation and risk management analyses for a cohesive $50B portfolio strategy at BlackRock.",
    system_prompt="""You are a senior investment strategist at BlackRock responsible for integrating detailed sector analyses into a comprehensive $50B portfolio allocation strategy. Your tasks include:
    - Synthesizing the fiscal, risk, and sector-specific insights provided by your fellow analysts.
    - Balancing the recommendations to yield clear guidance on both capital allocations and corresponding risk exposure for each market sector.
    - Formulating a unified strategy that specifies the optimal dollar and risk allocations per sector, ensuring that the overall portfolio adheres to BlackRock’s risk tolerance and performance objectives.
    - Delivering a final narrative that includes precise allocation figures, supported by tables and quantitative data.
    Ensure that your recommendations are actionable, data-driven, and well-aligned with institutional investment strategies.""",
    max_loops=1,
    model_name="groq/deepseek-r1-distill-qwen-32b",
    max_tokens=3000,
)

# Create majority voting system
majority_voting = MajorityVoting(
    name="Sector-Investment-Advisory-System",
    description="Multi-agent system for sector analysis that determines optimal capital and risk allocations for a $50B portfolio at BlackRock.",
    agents=agents,
    verbose=True,
    consensus_agent=consensus_agent,
)

# Run the analysis with majority voting
result = majority_voting.run(
    task=f"""Evaluate the current market sectors and determine the optimal allocation of a $50B portfolio for BlackRock. Your analysis should include:

1. A detailed table that outlines each sector along with the recommended dollar allocation and corresponding risk allocation.
2. A comprehensive review for each sector covering:
   - Fundamental performance metrics, historical trends, and growth outlook.
   - Quantitative risk assessments including volatility measures, stress test results, and risk-adjusted return evaluations.
3. Specific recommendations on how much capital (in dollars and as a percentage of $50B) should be invested in each sector.
4. A detailed explanation of the recommended risk allocation for each sector, ensuring the overall portfolio risk stays within acceptable thresholds.
5. A consolidated strategy that integrates both fiscal performance and risk management insights to support sector-based allocation decisions.
Provide your output with a clear structure, including descriptive sections and tables for clarity.


{macro_sector_analysis()}


"""
)

print(result)
