import base64
from pathlib import Path

from swarms import DecisionModel

# Reads GEMINI_API_KEY and calls Gemini's /v1beta/decisions endpoint.
model = DecisionModel(model_name="gemini-decision-1")

request = {
    "message": (
        "The blue mug I ordered on Monday arrived cracked on Wednesday. "
        "I'd like a replacement, not a refund."
    ),
    "order": {"id": "8821", "age_days": 2, "total_usd": 24.0},
    "customer": {"tier": "Gold", "returns_last_90_days": 0},
}

result = model.run(
    state=request,
    questions={
        "resolution": {
            "type": "choice",
            "instructions": "Which resolution fits this return request?",
            "criteria": {
                "replace": "Send the same item again.",
                "refund": "Return the payment to the customer.",
                "reject": "The request is outside the return policy.",
                "other": "Anything outside these outcomes.",
            },
        },
        "severity": {
            "type": "score",
            "instructions": "How badly is the item affected?",
            "criteria": [
                "Cosmetic: the item works as intended.",
                "Impaired: the item works with a workaround.",
                "Unusable: the item cannot be used.",
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
        "within_window": {
            "type": "noul",
            "instructions": "Is order.age_days 30 or less?",
        },
    },
)

answers = result["answers"]
resolution = answers["resolution"]

print(
    f"Resolution: {resolution['choice']} "
    f"(confidence {resolution['confidence']:.2f})"
)
print(f"Severity: {answers['severity']['score']:.2f} on a 0-2 scale")
print(f"Damaged: {answers['is_damaged']['noul']:.2f}")
print(f"Within window: {answers['within_window']['noul']:.2f}")

approved = (
    resolution["confidence"] >= 0.7
    and resolution["choice"] in ("replace", "refund")
    and answers["is_damaged"]["noul"] >= 0.8
    and answers["within_window"]["noul"] >= 0.8
)
print(
    f"Action: {resolution['choice']} automatically"
    if approved
    else "Action: send to a human"
)

# A list of content blocks carries text, images, audio, video and PDFs.
image_path = Path(__file__).parents[2] / "images" / "new_logo.png"
image_base64 = base64.b64encode(image_path.read_bytes()).decode(
    "ascii"
)

has_logo = model.noul(
    state=[
        {"type": "text", "text": "Photo attached to request #8821."},
        {
            "type": "image",
            "data": image_base64,
            "mime_type": "image/png",
        },
    ],
    instructions="Does the photo show a company logo?",
)
print(f"Photo shows a logo: {has_logo:.2f}")

print(f"Cost of both requests: {model.calculate_cost()}")
