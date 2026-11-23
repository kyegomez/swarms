from swarms import DecisionModel

# Reads GEMINI_API_KEY.
model = DecisionModel(model_name="gemini-decision-1")

probability = model.noul(
    state="The blue mug I ordered on Monday arrived cracked on Wednesday.",
    instructions="Did the item arrive damaged?",
)

print(f"Damaged: {probability:.2f}")
