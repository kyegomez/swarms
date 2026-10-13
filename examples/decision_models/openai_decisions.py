import base64
from pathlib import Path

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

answers = result["answers"]
print(
    f"Department: {answers['department']['choice']} "
    f"(confidence {answers['department']['confidence']:.2f})"
)
print(f"Severity: {answers['severity']['score']:.2f} on a 0-2 scale")
print(f"Urgency: {answers['is_urgent']['noul']:.2f}")

# Images must be inline base64 data URLs inside a list of user messages.
image_path = Path(__file__).parents[2] / "images" / "new_logo.png"
image_base64 = base64.b64encode(image_path.read_bytes()).decode(
    "ascii"
)

is_logo = model.noul(
    state=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Inspect this image."},
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{image_base64}",
                },
            ],
        }
    ],
    instructions="The image is a company logo, not a photograph.",
)
print(f"Logo probability: {is_logo:.2f}")

cost = model.calculate_cost()
print(
    f"{cost['input_tokens']} input tokens, ${cost['total_cost']:.6f}"
)
