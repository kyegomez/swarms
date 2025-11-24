"""
LLM Council Example: ETF Stock Analysis

This example demonstrates using the LLM Council to analyze ETF holdings
and provide stock investment recommendations.
"""

from swarms.structs.llm_council import LLMCouncil

# Create the council
council = LLMCouncil(verbose=True)

# ETF and stock analysis query
query = """
Analyze the top energy ETFs (including nuclear, solar, gas, and renewable energy)
and provide:
1. Top 5 best-performing energy stocks across all energy sectors
2. ETF recommendations for diversified energy exposure
3. Risk-return profiles for each recommendation
4. Current market conditions affecting energy investments
5. Allocation strategy for a $100,000 portfolio
6. Key metrics to track for each investment
"""

# Run the council
result = council.run(query)

# Print final response
print(result["final_response"])
