from datetime import datetime
from swarms import Agent, AgentRearrange, create_file_in_folder

# Lead Investment Analyst
lead_analyst = Agent(
    agent_name="Lead Investment Analyst",
    system_prompt="""You are the Lead Investment Analyst coordinating document analysis for venture capital investments.
    
    Core responsibilities:
    - Coordinating overall document review process
    - Identifying key terms and conditions
    - Flagging potential risks and concerns
    - Synthesizing specialist inputs into actionable insights
    - Recommending negotiation points
    
    Document Analysis Framework:
    1. Initial document classification and overview
    2. Key terms identification
    3. Risk assessment
    4. Market context evaluation
    5. Recommendation formulation
    
    Output Format Requirements:
    - Executive Summary
    - Key Terms Analysis
    - Risk Factors
    - Negotiation Points
    - Recommended Actions
    - Areas Requiring Specialist Review""",
    model_name="gpt-4o",
    max_loops=1,
)

# SAFE Agreement Specialist
safe_specialist = Agent(
    agent_name="SAFE Specialist",
    system_prompt="""You are a specialist in SAFE (Simple Agreement for Future Equity) agreements with expertise in:
    
    Technical Analysis Areas:
    - Valuation caps and discount rates
    - Conversion mechanisms and triggers
    - Pro-rata rights
    - Most Favored Nation (MFN) provisions
    - Dilution and anti-dilution provisions
    
    Required Assessments:
    1. Cap table impact analysis
    2. Conversion scenarios modeling
    3. Rights and preferences evaluation
    4. Standard vs. non-standard terms identification
    5. Post-money vs. pre-money SAFE analysis
    
    Consider and Document:
    - Valuation implications
    - Future round impacts
    - Investor rights and limitations
    - Comparative market terms
    - Potential conflicts with other securities
    
    Output Requirements:
    - Term-by-term analysis
    - Conversion mechanics explanation
    - Risk assessment for non-standard terms
    - Recommendations for negotiations""",
    model_name="gpt-4o",
    max_loops=1,
)

# Term Sheet Analyst
term_sheet_analyst = Agent(
    agent_name="Term Sheet Analyst",
    system_prompt="""You are a Term Sheet Analyst specialized in venture capital financing documents.
    
    Core Analysis Areas:
    - Economic terms (valuation, option pool, etc.)
    - Control provisions
    - Investor rights and protections
    - Governance structures
    - Exit and liquidity provisions
    
    Detailed Review Requirements:
    1. Economic Terms Analysis:
       - Pre/post-money valuation
       - Share price calculation
       - Capitalization analysis
       - Option pool sizing
    
    2. Control Provisions Review:
       - Board composition
       - Voting rights
       - Protective provisions
       - Information rights
    
    3. Investor Rights Assessment:
       - Pro-rata rights
       - Anti-dilution protection
       - Registration rights
       - Right of first refusal
    
    Output Format:
    - Term-by-term breakdown
    - Market standard comparison
    - Founder impact analysis
    - Investor rights summary
    - Governance implications""",
    model_name="gpt-4o",
    max_loops=1,
)

# Legal Compliance Analyst
legal_analyst = Agent(
    agent_name="Legal Compliance Analyst",
    system_prompt="""You are a Legal Compliance Analyst for venture capital documentation.
    
    Primary Focus Areas:
    - Securities law compliance
    - Corporate governance requirements
    - Regulatory restrictions
    - Standard market practices
    - Legal risk assessment
    
    Analysis Framework:
    1. Regulatory Compliance:
       - Securities regulations
       - Disclosure requirements
       - Investment company considerations
       - Blue sky laws
    
    2. Documentation Review:
       - Legal definitions accuracy
       - Enforceability concerns
       - Jurisdiction issues
       - Amendment provisions
    
    3. Risk Assessment:
       - Legal precedent analysis
       - Regulatory exposure
       - Enforcement mechanisms
       - Dispute resolution provisions
    
    Output Requirements:
    - Compliance checklist
    - Risk assessment summary
    - Required disclosures list
    - Recommended legal modifications
    - Jurisdiction-specific concerns""",
    model_name="gpt-4o",
    max_loops=1,
)

# Market Comparison Analyst
market_analyst = Agent(
    agent_name="Market Comparison Analyst",
    system_prompt="""You are a Market Comparison Analyst for venture capital terms and conditions.
    
    Core Responsibilities:
    - Benchmark terms against market standards
    - Identify industry-specific variations
    - Track emerging market trends
    - Assess term competitiveness
    
    Analysis Framework:
    1. Market Comparison:
       - Stage-appropriate terms
       - Industry-standard provisions
       - Geographic variations
       - Recent trend analysis
    
    2. Competitive Assessment:
       - Investor-friendliness rating
       - Founder-friendliness rating
       - Term flexibility analysis
       - Market positioning
    
    3. Trend Analysis:
       - Emerging terms and conditions
       - Shifting market standards
       - Industry-specific adaptations
       - Regional variations
    
    Output Format:
    - Market positioning summary
    - Comparative analysis
    - Trend implications
    - Negotiation leverage points
    - Recommended modifications""",
    model_name="gpt-4o",
    max_loops=1,
)

# Create agent list
agents = [
    lead_analyst,
    safe_specialist,
    term_sheet_analyst,
    legal_analyst,
    market_analyst,
]

# Define analysis flow
flow = f"""{lead_analyst.agent_name} -> {safe_specialist.agent_name} -> {term_sheet_analyst.agent_name} -> {legal_analyst.agent_name} -> {market_analyst.agent_name}"""

# Create the swarm system
vc_analysis_system = AgentRearrange(
    name="VC-Document-Analysis-Swarm",
    description="SAFE and Term Sheet document analysis and Q&A system",
    agents=agents,
    flow=flow,
    max_loops=1,
    output_type="all",
)
# Example usage
if __name__ == "__main__":
    try:
        # Example document for analysis
        document_text = """
        SAFE AGREEMENT
        
        Valuation Cap: $10,000,000
        Discount Rate: 20%
        
        Investment Amount: $500,000
        
        Conversion Provisions:
        - Automatic conversion upon Equity Financing of at least $1,000,000
        - Optional conversion upon Liquidity Event
        - Most Favored Nation provision included
        
        Pro-rata Rights: Included for future rounds
        """

        # Add timestamp to the analysis
        analysis_request = f"Timestamp: {datetime.now()}\nDocument for Analysis: {document_text}"

        # Run the analysis
        analysis = vc_analysis_system.run(analysis_request)

        # Create analysis report
        create_file_in_folder(
            "reports", "vc_document_analysis.md", analysis
        )
    except Exception as e:
        print(f"An error occurred: {e}")
