"""
LLM Council Example: Medical Diagnosis Analysis

This example demonstrates using the LLM Council to analyze symptoms
and provide diagnostic insights.
"""

from swarms.structs.llm_council import LLMCouncil

# Create the council
council = LLMCouncil(verbose=True)

# Medical diagnosis query
query = """
A 35-year-old patient presents with:
- Persistent fatigue for 3 months
- Unexplained weight loss (15 lbs)
- Night sweats
- Intermittent low-grade fever
- Swollen lymph nodes in neck and armpits
- Recent blood work shows elevated ESR and CRP

Provide:
1. Differential diagnosis with most likely conditions ranked
2. Additional diagnostic tests needed to confirm
3. Red flag symptoms requiring immediate attention
4. Possible causes and risk factors
5. Recommended next steps for the patient
6. When to seek emergency care
"""

# Run the council
result = council.run(query)

# Print final response
print(result["final_response"])
