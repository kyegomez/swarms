from swarms.structs.llm_council import LLMCouncil

# Create the council
council = LLMCouncil(verbose=True, output_type="final")

# Example query
query = "What are the top five best energy stocks across nuclear, solar, gas, and other energy sources?"

# Run the council
result = council.run(query)

print(result)
