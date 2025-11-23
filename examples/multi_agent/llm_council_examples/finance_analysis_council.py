"""
LLM Council Example: Financial Analysis

This example demonstrates using the LLM Council to provide comprehensive
financial analysis and investment recommendations.
"""

from swarms.structs.llm_council import LLMCouncil

# Create the council
council = LLMCouncil(verbose=True)

# Financial analysis query
query = """
Provide a comprehensive financial analysis for investing in emerging markets 
technology ETFs. Include:
1. Risk assessment and volatility analysis
2. Historical performance trends
3. Sector composition and diversification benefits
4. Comparison with developed market tech ETFs
5. Recommended allocation percentage for a moderate risk portfolio
6. Key factors to monitor going forward
"""

# Run the council
result = council.run(query)

# Print final response
print(result["final_response"])

