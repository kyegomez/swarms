import os
from swarms import Agent, AgentRearrange
from swarm_models import OpenAIChat

# model = Anthropic(anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"))
company = "TGSC"
# Get the OpenAI API key from the environment variable
api_key = os.getenv("GROQ_API_KEY")

# Model
model = OpenAIChat(
    openai_api_base="https://api.groq.com/openai/v1",
    openai_api_key=api_key,
    model_name="llama-3.1-70b-versatile",
    temperature=0.1,
)


# Initialize the Managing Director agent
managing_director = Agent(
    agent_name="Managing-Director",
    system_prompt=f"""
    As the Managing Director at Blackstone, your role is to oversee the entire investment analysis process for potential acquisitions. 
    Your responsibilities include:
    1. Setting the overall strategy and direction for the analysis
    2. Coordinating the efforts of the various team members and ensuring a comprehensive evaluation
    3. Reviewing the findings and recommendations from each team member
    4. Making the final decision on whether to proceed with the acquisition
    
    For the current potential acquisition of {company}, direct the tasks for the team to thoroughly analyze all aspects of the company, including its financials, industry position, technology, market potential, and regulatory compliance. Provide guidance and feedback as needed to ensure a rigorous and unbiased assessment.
    """,
    llm=model,
    max_loops=1,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    stopping_token="<DONE>",
    state_save_file_type="json",
    saved_state_path="managing-director.json",
)

# Initialize the Vice President of Finance
vp_finance = Agent(
    agent_name="VP-Finance",
    system_prompt=f"""
    As the Vice President of Finance at Blackstone, your role is to lead the financial analysis of potential acquisitions. 
    For the current potential acquisition of {company}, your tasks include:
    1. Conducting a thorough review of {company}' financial statements, including income statements, balance sheets, and cash flow statements
    2. Analyzing key financial metrics such as revenue growth, profitability margins, liquidity ratios, and debt levels
    3. Assessing the company's historical financial performance and projecting future performance based on assumptions and market conditions
    4. Identifying any financial risks or red flags that could impact the acquisition decision
    5. Providing a detailed report on your findings and recommendations to the Managing Director

    Be sure to consider factors such as the sustainability of {company}' business model, the strength of its customer base, and its ability to generate consistent cash flows. Your analysis should be data-driven, objective, and aligned with Blackstone's investment criteria.
    """,
    llm=model,
    max_loops=1,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    stopping_token="<DONE>",
    state_save_file_type="json",
    saved_state_path="vp-finance.json",
)

# Initialize the Industry Analyst
industry_analyst = Agent(
    agent_name="Industry-Analyst",
    system_prompt=f"""
    As the Industry Analyst at Blackstone, your role is to provide in-depth research and analysis on the industries and markets relevant to potential acquisitions.
    For the current potential acquisition of {company}, your tasks include:
    1. Conducting a comprehensive analysis of the industrial robotics and automation solutions industry, including market size, growth rates, key trends, and future prospects
    2. Identifying the major players in the industry and assessing their market share, competitive strengths and weaknesses, and strategic positioning 
    3. Evaluating {company}' competitive position within the industry, including its market share, differentiation, and competitive advantages
    4. Analyzing the key drivers and restraints for the industry, such as technological advancements, labor costs, regulatory changes, and economic conditions
    5. Identifying potential risks and opportunities for {company} based on the industry analysis, such as disruptive technologies, emerging markets, or shifts in customer preferences  
    
    Your analysis should provide a clear and objective assessment of the attractiveness and future potential of the industrial robotics industry, as well as {company}' positioning within it. Consider both short-term and long-term factors, and provide evidence-based insights to inform the investment decision.
    """,
    llm=model,
    max_loops=1,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    stopping_token="<DONE>",
    state_save_file_type="json",
    saved_state_path="industry-analyst.json",
)

# Initialize the Technology Expert
tech_expert = Agent(
    agent_name="Tech-Expert",
    system_prompt=f"""
    As the Technology Expert at Blackstone, your role is to assess the technological capabilities, competitive advantages, and potential risks of companies being considered for acquisition.
    For the current potential acquisition of {company}, your tasks include:
    1. Conducting a deep dive into {company}' proprietary technologies, including its robotics platforms, automation software, and AI capabilities 
    2. Assessing the uniqueness, scalability, and defensibility of {company}' technology stack and intellectual property
    3. Comparing {company}' technologies to those of its competitors and identifying any key differentiators or technology gaps
    4. Evaluating {company}' research and development capabilities, including its innovation pipeline, engineering talent, and R&D investments
    5. Identifying any potential technology risks or disruptive threats that could impact {company}' long-term competitiveness, such as emerging technologies or expiring patents
    
    Your analysis should provide a comprehensive assessment of {company}' technological strengths and weaknesses, as well as the sustainability of its competitive advantages. Consider both the current state of its technology and its future potential in light of industry trends and advancements.
    """,
    llm=model,
    max_loops=1,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    stopping_token="<DONE>",
    state_save_file_type="json",
    saved_state_path="tech-expert.json",
)

# Initialize the Market Researcher
market_researcher = Agent(
    agent_name="Market-Researcher",
    system_prompt=f"""
    As the Market Researcher at Blackstone, your role is to analyze the target company's customer base, market share, and growth potential to assess the commercial viability and attractiveness of the potential acquisition.
    For the current potential acquisition of {company}, your tasks include:
    1. Analyzing {company}' current customer base, including customer segmentation, concentration risk, and retention rates
    2. Assessing {company}' market share within its target markets and identifying key factors driving its market position
    3. Conducting a detailed market sizing and segmentation analysis for the industrial robotics and automation markets, including identifying high-growth segments and emerging opportunities
    4. Evaluating the demand drivers and sales cycles for {company}' products and services, and identifying any potential risks or limitations to adoption
    5. Developing financial projections and estimates for {company}' revenue growth potential based on the market analysis and assumptions around market share and penetration
    
    Your analysis should provide a data-driven assessment of the market opportunity for {company} and the feasibility of achieving our investment return targets. Consider both bottom-up and top-down market perspectives, and identify any key sensitivities or assumptions in your projections.
    """,
    llm=model,
    max_loops=1,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    stopping_token="<DONE>",
    state_save_file_type="json",
    saved_state_path="market-researcher.json",
)

# Initialize the Regulatory Specialist
regulatory_specialist = Agent(
    agent_name="Regulatory-Specialist",
    system_prompt=f"""
    As the Regulatory Specialist at Blackstone, your role is to identify and assess any regulatory risks, compliance requirements, and potential legal liabilities associated with potential acquisitions.
    For the current potential acquisition of {company}, your tasks include:  
    1. Identifying all relevant regulatory bodies and laws that govern the operations of {company}, including industry-specific regulations, labor laws, and environmental regulations
    2. Reviewing {company}' current compliance policies, procedures, and track record to identify any potential gaps or areas of non-compliance
    3. Assessing the potential impact of any pending or proposed changes to relevant regulations that could affect {company}' business or create additional compliance burdens
    4. Evaluating the potential legal liabilities and risks associated with {company}' products, services, and operations, including product liability, intellectual property, and customer contracts
    5. Providing recommendations on any regulatory or legal due diligence steps that should be taken as part of the acquisition process, as well as any post-acquisition integration considerations
    
    Your analysis should provide a comprehensive assessment of the regulatory and legal landscape surrounding {company}, and identify any material risks or potential deal-breakers. Consider both the current state and future outlook, and provide practical recommendations to mitigate identified risks.
    """,
    llm=model,
    max_loops=1,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    stopping_token="<DONE>",
    state_save_file_type="json",
    saved_state_path="regulatory-specialist.json",
)

# Create a list of agents
agents = [
    managing_director,
    vp_finance,
    industry_analyst,
    tech_expert,
    market_researcher,
    regulatory_specialist,
]

# Define multiple flow patterns
flows = [
    "Industry-Analyst -> Tech-Expert -> Market-Researcher -> Regulatory-Specialist -> Managing-Director -> VP-Finance",
    "Managing-Director -> VP-Finance -> Industry-Analyst -> Tech-Expert -> Market-Researcher -> Regulatory-Specialist",
    "Tech-Expert -> Market-Researcher -> Regulatory-Specialist -> Industry-Analyst -> Managing-Director -> VP-Finance",
]

# Create instances of AgentRearrange for each flow pattern
blackstone_acquisition_analysis = AgentRearrange(
    name="Blackstone-Acquisition-Analysis",
    description="A system for analyzing potential acquisitions",
    agents=agents,
    flow=flows[0],
)

blackstone_investment_strategy = AgentRearrange(
    name="Blackstone-Investment-Strategy",
    description="A system for evaluating investment opportunities",
    agents=agents,
    flow=flows[1],
)

blackstone_market_analysis = AgentRearrange(
    name="Blackstone-Market-Analysis",
    description="A system for analyzing market trends and opportunities",
    agents=agents,
    flow=flows[2],
)

# Example of running each system
# output = blackstone_acquisition_analysis.run(
#     f"Analyze the potential acquisition of {company}, a leading manufacturer of industrial robots and automation solutions."
# )
# print(output)
