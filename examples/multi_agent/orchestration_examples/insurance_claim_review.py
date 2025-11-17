from swarms import Agent
from swarms.structs.multi_agent_debates import PeerReviewProcess

# Initialize the insurance claim reviewers and author
claims_adjuster = Agent(
    agent_name="Claims-Adjuster",
    agent_description="Senior claims adjuster with expertise in complex medical claims",
    system_prompt="""You are a senior claims adjuster specializing in:
    - Complex medical claims evaluation
    - Policy coverage analysis
    - Claims documentation review
    - Fraud detection
    - Regulatory compliance
    
    Review claims thoroughly and provide detailed assessments based on policy terms and medical necessity.""",
    model_name="claude-3-sonnet-20240229",
)

medical_director = Agent(
    agent_name="Medical-Director",
    agent_description="Insurance medical director for clinical review",
    system_prompt="""You are an insurance medical director expert in:
    - Clinical necessity evaluation
    - Treatment protocol assessment
    - Medical cost analysis
    - Quality of care review
    
    Evaluate medical aspects of claims and ensure appropriate healthcare delivery.""",
    model_name="claude-3-sonnet-20240229",
)

legal_specialist = Agent(
    agent_name="Legal-Specialist",
    agent_description="Insurance legal specialist for compliance review",
    system_prompt="""You are an insurance legal specialist focusing on:
    - Regulatory compliance
    - Policy interpretation
    - Legal risk assessment
    - Documentation requirements
    
    Review claims for legal compliance and policy adherence.""",
    model_name="claude-3-sonnet-20240229",
)

claims_processor = Agent(
    agent_name="Claims-Processor",
    agent_description="Claims processor who submitted the initial claim",
    system_prompt="""You are a claims processor responsible for:
    - Initial claim submission
    - Documentation gathering
    - Policy verification
    - Benefit calculation
    
    Present claims clearly and respond to reviewer feedback.""",
    model_name="claude-3-sonnet-20240229",
)

# Initialize the peer review process
review_process = PeerReviewProcess(
    reviewers=[claims_adjuster, medical_director, legal_specialist],
    author=claims_processor,
    review_rounds=2,
    output_type="str-all-except-first",
)

# Complex claim case for review
claim_case = """
High-Value Claim Review Required:
Patient underwent emergency TAVR (Transcatheter Aortic Valve Replacement) at out-of-network facility
while traveling. Claim value: $285,000

Key Elements for Review:
1. Emergency nature verification
2. Out-of-network coverage applicability
3. Procedure medical necessity
4. Pricing comparison with in-network facilities
5. Patient's policy coverage limits
6. Network adequacy requirements
7. State regulatory compliance

Additional Context:
- Patient has comprehensive coverage with out-of-network benefits
- Procedure was performed without prior authorization
- Local in-network facilities were 200+ miles away
- Patient was stabilized but required urgent intervention within 24 hours
"""

# Execute the review process
review_output = review_process.run(claim_case)
print(review_output)
