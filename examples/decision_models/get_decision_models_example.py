from swarms import (
    DecisionModel,
    get_decision_model_prices,
    get_decision_models,
)

# Fetched live from each provider whose keys are in your environment or .env.
models = get_decision_models()
prices = get_decision_model_prices()
print("Supported decision models (US dollars per million tokens):")
for model_name in models:
    price = prices.get(model_name)
    if price is None:
        print(f"  {model_name}: price unknown")
    else:
        print(
            f"  {model_name}: ${price['input']} input, ${price['output']} output"
        )
print()

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
    usage = model.usage
    line = (
        f"{model_name}: urgency {urgency:.2f}, "
        f"{usage['input_tokens']} input + {usage['output_tokens']} output tokens"
    )
    if model_name in prices:
        line += f", ${model.calculate_cost()['total_cost']:.6f}"
    print(line)
