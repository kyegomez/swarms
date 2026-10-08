"""
LLM Council Example: ETF Stock Analysis

This example demonstrates using the LLM Council to analyze ETF holdings
and provide stock investment recommendations.
"""

from swarms import LLMCouncil

# Create the council
council = LLMCouncil(verbose=True, output_type="final")

# ETF and stock analysis task
task = """
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
result = council.run(task=task)

# Print final response
print(result)
