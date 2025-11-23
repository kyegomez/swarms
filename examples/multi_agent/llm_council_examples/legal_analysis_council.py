"""
LLM Council Example: Legal Analysis

This example demonstrates using the LLM Council to analyze legal scenarios
and provide comprehensive legal insights.
"""

from swarms.structs.llm_council import LLMCouncil

# Create the council
council = LLMCouncil(verbose=True)

# Legal analysis query
query = """
A startup is considering using AI-generated content for their marketing materials.
Analyze the legal implications including:
1. Intellectual property rights and ownership of AI-generated content
2. Copyright and trademark considerations
3. Liability for AI-generated content that may be inaccurate or misleading
4. Compliance with advertising regulations (FTC, FDA, etc.)
5. Data privacy implications if using customer data to train models
6. Contractual considerations with AI service providers
7. Risk mitigation strategies
8. Best practices for legal compliance
"""

# Run the council
result = council.run(query)

# Print final response
print(result["final_response"])

