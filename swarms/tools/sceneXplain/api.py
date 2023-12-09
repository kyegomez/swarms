import requests
import json
from ..tool import Tool
import os


def build_tool(config) -> Tool:
    tool = Tool(
        "Image Explainer",
        "Tool that adds the capability to explain images.",
        name_for_model="Image Explainer",
        description_for_model=(
            "An Image Captioning Tool: Use this tool to generate a detailed caption "
            "for an image. The input can be an image file of any format, and "
            "the output will be a text description that covers every detail of the image."
        ),
        logo_url="https://scenex.jina.ai/SceneX%20-%20Light.svg",
        contact_email="hello@contact.com",
        legal_info_url="hello@legal.com",
    )

    scenex_api_key = config["subscription_key"]
    scenex_api_url: str = (
        "https://us-central1-causal-diffusion.cloudfunctions.net/describe"
    )

    @tool.get("/describe_image")
    def describe_image(image: str):
        """Get the text description of an image."""
        headers = {
            "x-api-key": f"token {scenex_api_key}",
            "content-type": "application/json",
        }
        payload = {
            "data": [
                {
                    "image": image,
                    "algorithm": "Ember",
                    "languages": ["en"],
                }
            ]
        }
        response = requests.post(scenex_api_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json().get("result", [])
        img = result[0] if result else {}
        description = img.get("text", "")
        if not description:
            return "No description found."

        return description

    return tool
