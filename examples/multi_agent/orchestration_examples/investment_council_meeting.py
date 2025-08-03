from swarms import Agent
from swarms.structs.multi_agent_debates import CouncilMeeting

# Initialize the investment council members
investment_chair = Agent(
    agent_name="Investment-Chair",
    agent_description="Investment committee chairperson",
    system_prompt="""You are the investment committee chair with expertise in:
    - Portfolio strategy
    - Risk management
    - Asset allocation
    - Investment governance
    - Performance oversight
    
    Lead the council meeting effectively while ensuring thorough analysis and proper decision-making.""",
    model_name="claude-3-sonnet-20240229",
)

equity_strategist = Agent(
    agent_name="Equity-Strategist",
    agent_description="Global equity investment strategist",
    system_prompt="""You are a senior equity strategist specializing in:
    - Global equity markets
    - Sector allocation
    - Factor investing
    - ESG integration
    - Market analysis
    
    Provide insights on equity investment opportunities and risks.""",
    model_name="claude-3-sonnet-20240229",
)

fixed_income_specialist = Agent(
    agent_name="Fixed-Income-Specialist",
    agent_description="Fixed income portfolio manager",
    system_prompt="""You are a fixed income specialist expert in:
    - Bond market analysis
    - Credit risk assessment
    - Duration management
    - Yield curve strategies
    - Fixed income derivatives
    
    Contribute expertise on fixed income markets and strategies.""",
    model_name="claude-3-sonnet-20240229",
)

risk_manager = Agent(
    agent_name="Risk-Manager",
    agent_description="Investment risk management specialist",
    system_prompt="""You are a risk management expert focusing on:
    - Portfolio risk analysis
    - Risk modeling
    - Scenario testing
    - Risk budgeting
    - Compliance oversight
    
    Provide risk assessment and mitigation strategies.""",
    model_name="claude-3-sonnet-20240229",
)

alternatives_expert = Agent(
    agent_name="Alternatives-Expert",
    agent_description="Alternative investments specialist",
    system_prompt="""You are an alternative investments expert specializing in:
    - Private equity
    - Real estate
    - Hedge funds
    - Infrastructure
    - Private credit
    
    Contribute insights on alternative investment opportunities.""",
    model_name="claude-3-sonnet-20240229",
)

# Initialize the council meeting
council = CouncilMeeting(
    council_members=[
        equity_strategist,
        fixed_income_specialist,
        risk_manager,
        alternatives_expert,
    ],
    chairperson=investment_chair,
    voting_rounds=2,
    require_consensus=True,
    output_type="str-all-except-first",
)

# Investment proposal for discussion
investment_proposal = """
Strategic Asset Allocation Review and Proposal

Current Market Context:
- Rising inflation expectations
- Monetary policy tightening cycle
- Geopolitical tensions
- ESG considerations
- Private market opportunities

Proposed Changes:
1. Reduce developed market equity allocation by 5%
2. Increase private credit allocation by 3%
3. Add 2% to infrastructure investments
4. Implement ESG overlay across equity portfolio
5. Extend fixed income duration

Risk Considerations:
- Impact on portfolio liquidity
- Currency exposure
- Interest rate sensitivity
- Manager selection risk
- ESG implementation challenges

Required Decisions:
1. Approve/modify allocation changes
2. Set implementation timeline
3. Define risk monitoring framework
4. Establish performance metrics
5. Determine rebalancing triggers
"""

# Execute the council meeting
council_output = council.run(investment_proposal)
print(council_output)
