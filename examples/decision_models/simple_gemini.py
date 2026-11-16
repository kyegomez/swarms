from swarms import DecisionModel

# Reads GEMINI_API_KEY and calls Gemini's /v1beta/decisions endpoint.
model = DecisionModel(model_name="gemini-decision-1")

result = model.run(
    state="The blue mug I ordered on Monday arrived cracked on Wednesday.",
    questions={
        "queue": {
            "type": "choice",
            "instructions": "Which support queue should handle this message?",
            "criteria": {
                "billing": "Payment, invoice, or refund questions.",
                "replacements": "Damaged, missing, or wrong items.",
                "general": "Product questions and account updates.",
            },
        },
        "severity": {
            "type": "score",
            "instructions": "How severe is this issue for the customer?",
            "criteria": [
                "Minor: the item is usable as it is.",
                "Moderate: the item is usable with a workaround.",
                "Severe: the item cannot be used.",
            ],
        },
        "is_damaged": {
            "type": "noul",
            "instructions": "Did the item arrive damaged?",
            "criteria": {
                "true": "The item itself is broken or faulty.",
                "false": "A preference, sizing, shipping or pricing complaint.",
            },
        },
    },
)

print(result)
print(model.calculate_cost())
