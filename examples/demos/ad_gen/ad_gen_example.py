import os
import random

from dotenv import load_dotenv

from swarms.models import OpenAIChat
from swarms.models.stable_diffusion import StableDiffusion
from swarms.structs import Agent

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
stability_api_key = os.getenv("STABILITY_API_KEY")

# Initialize the language model and image generation model
llm = OpenAIChat(
    openai_api_key=openai_api_key, temperature=0.5, max_tokens=3000
)
sd_api = StableDiffusion(api_key=stability_api_key)


# Creative Concept Generator for Product Ads
class ProductAdConceptGenerator:
    def __init__(self, product_name):
        self.product_name = product_name
        self.themes = [
            "futuristic",
            "rustic",
            "luxurious",
            "minimalistic",
            "vibrant",
            "elegant",
            "retro",
            "urban",
            "ethereal",
            "surreal",
            "artistic",
            "tech-savvy",
            "vintage",
            "natural",
            "sophisticated",
            "playful",
            "dynamic",
            "serene",
            "lasers,lightning",
        ]
        self.contexts = [
            "in an everyday setting",
            "in a rave setting",
            "in an abstract environment",
            "in an adventurous context",
            "surrounded by nature",
            "in a high-tech setting",
            "in a historical context",
            "in a busy urban scene",
            "in a tranquil and peaceful setting",
            "against a backdrop of city lights",
            "in a surreal dreamscape",
            "in a festive atmosphere",
            "in a luxurious setting",
            "in a playful and colorful background",
            "in an ice cave setting",
            "in a serene and calm landscape",
        ]
        self.contexts = [
            "high realism product ad (extremely creative)"
        ]

    def generate_concept(self):
        theme = random.choice(self.themes)
        context = random.choice(self.contexts)
        return (
            f"{theme} inside a {style} {self.product_name}, {context}"
        )


# User input
product_name = input(
    "Enter a product name for ad creation (e.g., 'PS5', 'AirPods',"
    " 'Kirkland Vodka'): "
)

# Generate creative concept
concept_generator = ProductAdConceptGenerator(product_name)
creative_concept = concept_generator.generate_concept()

# Generate product image based on the creative concept
image_paths = sd_api.run(creative_concept)

# Generate ad copy
ad_copy_agent = Agent(llm=llm, max_loops=1)
ad_copy_prompt = (
    f"Write a compelling {social_media_platform} ad copy for a"
    f" product photo showing {product_name} {creative_concept}."
)
ad_copy = ad_copy_agent.run(task=ad_copy_prompt)

# Output the results
print("Creative Concept:", concept_result)
print("Design Ideas:", design_result)
print("Ad Copy:", copywriting_result)
print(
    "Image Path:",
    image_paths[0] if image_paths else "No image generated",
)
