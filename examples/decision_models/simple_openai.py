from swarms import DecisionModel

# Reads OPENAI_API_KEY and calls OpenAI's /v1/decisions endpoint.
model = DecisionModel(model_name="gpt-6-luna")

result = model.run(
    state="Export fails in Safari but works in Chrome. I was charged twice for my order.",
    questions={
        "department": {
            "type": "choice",
            "instructions": "Which department should handle this complaint?",
            "criteria": {
                "billing": "Payments, invoices, and refunds.",
                "technical": "Problems using the product.",
                "shipping": "Delivery and tracking.",
                "other": "Requests outside these categories.",
            },
        },
        "severity": {
            "type": "score",
            "instructions": "How severe is this issue?",
            "criteria": [
                "Cosmetic: appearance only, no lost functionality.",
                "Workaround available: a task fails, but another way works.",
                "Fully blocked: a task fails with no workaround.",
            ],
        },
        "is_urgent": {
            "type": "noul",
            "instructions": "The customer needs a reply today.",
        },
    },
)

print(result)
