"""
LLM Council Example: Research Analysis

This example demonstrates using the LLM Council to conduct comprehensive
research analysis on complex topics.
"""

from swarms.structs.llm_council import LLMCouncil

# Create the council
council = LLMCouncil(verbose=True)

# Research analysis query
query = """
Conduct a comprehensive analysis of the potential impact of climate change 
on global food security over the next 20 years. Include:
1. Key climate factors affecting agriculture (temperature, precipitation, extreme weather)
2. Regional vulnerabilities and impacts on major food-producing regions
3. Crop yield projections and food availability scenarios
4. Economic implications and food price volatility
5. Adaptation strategies and technological solutions
6. Policy recommendations for governments and international organizations
7. Role of innovation in agriculture (precision farming, GMOs, vertical farming)
8. Social and geopolitical implications of food insecurity
"""

# Run the council
result = council.run(query)

# Print final response
print(result["final_response"])

