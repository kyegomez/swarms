from swarms.structs.llm_council import LLMCouncil

# Example usage of the LLM Council without a function:
# Create the council
council = LLMCouncil(verbose=True)

# Example query
query = "What are the top five best energy stocks across nuclear, solar, gas, and other energy sources?"

# Run the council
result = council.run(query)

# Print final response
print(result["final_response"])

# Optionally print evaluations
print("\n\n" + "="*80)
print("EVALUATIONS")
print("="*80)
for name, evaluation in result["evaluations"].items():
    print(f"\n{name}:")
    print(evaluation[:500] + "..." if len(evaluation) > 500 else evaluation)

