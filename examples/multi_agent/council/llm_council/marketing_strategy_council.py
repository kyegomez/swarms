"""
LLM Council Example: Marketing Strategy Analysis

This example demonstrates using the LLM Council to analyze and develop
comprehensive marketing strategies by leveraging multiple AI perspectives.
"""

from swarms.structs.llm_council import LLMCouncil

# Create the council
council = LLMCouncil(verbose=True)

# Marketing strategy query
query = """
Analyze the marketing strategy for a new sustainable energy startup launching 
a solar panel subscription service. Provide recommendations on:
1. Target audience segmentation
2. Key messaging and value propositions
3. Marketing channels and budget allocation
4. Competitive positioning
5. Launch timeline and milestones
"""

# Run the council
result = council.run(query)

# Print final response
print(result["final_response"])
