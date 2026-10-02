from swarms import DecisionModel, get_decision_models

# Fetched live from each provider whose keys are in your environment or .env.
models = get_decision_models()
print(f"Supported decision models: {models}\n")

ticket = (
    "Checkout has been failing for every customer for the last hour."
)

for model_name in models:
    try:
        model = DecisionModel(model_name=model_name)
    except ValueError as error:
        print(f"{model_name}: skipped, {error}")
        continue

    urgency = model.noul(
        state=ticket,
        instructions="Is this support request urgent?",
    )
    print(f"{model_name}: urgency {urgency:.2f}")
