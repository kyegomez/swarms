"""
LLM Council Example: Medical Treatment Analysis

This example demonstrates using the LLM Council to analyze medical treatments
and provide comprehensive treatment recommendations.
"""

from swarms.structs.llm_council import LLMCouncil

# Create the council
council = LLMCouncil(verbose=True)

# Medical treatment query
query = """
A 45-year-old patient with Type 2 diabetes, hypertension, and early-stage 
kidney disease needs treatment recommendations. Provide:
1. Comprehensive treatment plan addressing all conditions
2. Medication options with pros/cons for each condition
3. Lifestyle modifications and their expected impact
4. Monitoring schedule and key metrics to track
5. Potential drug interactions and contraindications
6. Expected outcomes and timeline for improvement
7. When to consider specialist referrals
"""

# Run the council
result = council.run(query)

# Print final response
print(result["final_response"])

