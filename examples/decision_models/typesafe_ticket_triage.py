from swarms import DecisionModel

# Reads TYPESAFE_API_KEY and calls TypeSafe's jev-latest by default.
model = DecisionModel()

ticket = {
    "message": (
        "Hi, I've been trying to connect my Stripe account for 3 days "
        "and the integration keeps failing. I'm losing sales. "
        "Please help ASAP."
    ),
    "customer_plan": "enterprise",
}

result = model.run(
    state=ticket,
    questions={
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this ticket?",
            "criteria": {
                "billing": "Payment or subscription issues",
                "technical": "Bugs or integration problems",
                "sales": "Pricing or account questions",
            },
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated does the customer appear?",
            "criteria": [
                "Calm, just stating facts",
                "Frustrated but civil",
                "Very angry, strong language",
            ],
        },
        "is_urgent": {
            "type": "noul",
            "instructions": "The message conveys urgency or time-sensitivity.",
        },
    },
)

answers = result["answers"]
department = answers["department"]
frustration = answers["frustration"]["score"]
urgency = answers["is_urgent"]["noul"]

print(f"Answered by {result['model']}")
print(
    f"Department: {department['choice']} "
    f"(confidence {department['confidence']:.2f})"
)
print(f"Probabilities: {department['probabilities']}")
print(f"Frustration: {frustration:.2f} on a 0-2 scale")
print(f"Urgency: {urgency:.2f}")

if department["confidence"] < 0.6:
    print(
        "Action: send to a human, the model is unsure who owns this."
    )
elif urgency > 0.8 and frustration >= 1:
    print(
        f"Action: page the {department['choice']} on-call engineer."
    )
else:
    print(f"Action: queue for the {department['choice']} team.")
