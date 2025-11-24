"""
LLM Council Example: Business Strategy Development

This example demonstrates using the LLM Council to develop comprehensive
business strategies for new ventures.
"""

from swarms.structs.llm_council import LLMCouncil

# Create the council
council = LLMCouncil(verbose=True)

# Business strategy query
query = """
A tech startup wants to launch an AI-powered personal finance app targeting 
millennials and Gen Z. Develop a comprehensive business strategy including:
1. Market opportunity and competitive landscape analysis
2. Product positioning and unique value proposition
3. Go-to-market strategy and customer acquisition plan
4. Revenue model and pricing strategy
5. Key partnerships and distribution channels
6. Resource requirements and funding needs
7. Risk assessment and mitigation strategies
8. Success metrics and KPIs for first 12 months
"""

# Run the council
result = council.run(query)

# Print final response
print(result["final_response"])
