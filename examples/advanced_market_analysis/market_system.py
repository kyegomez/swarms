
from swarms.structs.agent import Agent
from swarms.prompts.finance_agent_sys_prompt import FINANCIAL_AGENT_SYS_PROMPT

# Technical Analysis Specialist
technical_analyst = Agent(
    agent_name="Technical-Analysis-Expert",
    agent_description="Advanced technical analysis specialist focusing on complex market patterns",
    system_prompt="""You are an expert Technical Analyst specializing in:
    1. Advanced Pattern Recognition (Elliot Wave, Wyckoff Method)
    2. Multi-timeframe Analysis
    3. Volume Profile Analysis
    4. Market Structure Analysis
    5. Intermarket Analysis""",
    max_loops=3,
    model_name="gpt-4"
)

# Fundamental Analysis Expert
fundamental_analyst = Agent(
    agent_name="Fundamental-Analysis-Expert",
    agent_description="Deep fundamental analysis specialist",
    system_prompt="""You are a Fundamental Analysis expert focusing on:
    1. Advanced Financial Statement Analysis
    2. Economic Indicator Impact Assessment
    3. Industry Competitive Analysis
    4. Global Macro Trends
    5. Corporate Governance Evaluation""",
    max_loops=3,
    model_name="gpt-4"
)

# Risk Management Specialist
risk_analyst = Agent(
    agent_name="Risk-Management-Expert",
    agent_description="Complex risk analysis and management specialist",
    system_prompt="""You are a Risk Management expert specializing in:
    1. Portfolio Risk Assessment
    2. Value at Risk (VaR) Analysis
    3. Stress Testing Scenarios
    4. Correlation Analysis
    5. Risk-Adjusted Performance Metrics""",
    max_loops=3,
    model_name="gpt-4"
)

class MarketAnalysisSystem:
    def __init__(self):
        self.agents = [technical_analyst, fundamental_analyst, risk_analyst]
        
    def comprehensive_analysis(self, asset_data):
        analysis_results = []
        for agent in self.agents:
            analysis = agent.run(f"Analyze this asset data: {asset_data}")
            analysis_results.append({
                "analyst": agent.agent_name,
                "analysis": analysis
            })
        
        # Synthesize results through risk analyst for final recommendation
        final_analysis = risk_analyst.run(
            f"Synthesize these analyses and provide a final recommendation: {analysis_results}"
        )
        
        return {
            "detailed_analysis": analysis_results,
            "final_recommendation": final_analysis
        }

# Usage
analysis_system = MarketAnalysisSystem()
