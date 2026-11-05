from swarms import LLMCouncil

# Create the council
council = LLMCouncil(verbose=True, output_type="final")

# Example task
task = "What are the top five best energy stocks across nuclear, solar, gas, and other energy sources?"

# Run the council
result = council.run(task=task)

print(result)
