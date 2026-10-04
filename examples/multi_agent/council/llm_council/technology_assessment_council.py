"""
LLM Council Example: Technology Assessment

This example demonstrates using the LLM Council to assess emerging technologies
and their business implications.
"""

from swarms import LLMCouncil

# Create the council
council = LLMCouncil(verbose=True, output_type="final")

# Technology assessment task
task = """
Evaluate the business potential and implementation strategy for integrating 
quantum computing capabilities into a financial services company. Consider:
1. Current state of quantum computing technology
2. Specific use cases in financial services (risk modeling, portfolio optimization, fraud detection)
3. Competitive advantages and potential ROI
4. Implementation timeline and resource requirements
5. Technical challenges and limitations
6. Risk factors and mitigation strategies
7. Partnership opportunities with quantum computing providers
8. Expected timeline for practical business value
"""

# Run the council
result = council.run(task=task)

# Print final response
print(result)
