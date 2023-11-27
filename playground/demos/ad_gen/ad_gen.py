import random
import os
from dotenv import load_dotenv
from swarms.models import OpenAIChat
from playground.models.stable_diffusion import StableDiffusion
from swarms.structs import Flow, SequentialWorkflow

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
stability_api_key = os.getenv("STABILITY_API_KEY")

# Initialize the language model and image generation model
llm = OpenAIChat(openai_api_key=openai_api_key, temperature=0.5, max_tokens=3000)
sd_api = StableDiffusion(api_key=stability_api_key)

def run_task(description, product_name, flow, **kwargs):
    full_description = f"{description} about {product_name}"  # Incorporate product name into the task
    result = flow.run(task=full_description, **kwargs)
    return result


# Creative Concept Generator
class ProductPromptGenerator:
    def __init__(self, product_name):
        self.product_name = product_name
        self.themes = ["lightning", "sunset", "ice cave", "space", "forest", "ocean", "mountains", "cityscape"]
        self.styles = ["translucent", "floating in mid-air", "expanded into pieces", "glowing", "mirrored", "futuristic"]
        self.contexts = ["high realism product ad (extremely creative)"]

    def generate_prompt(self):
        theme = random.choice(self.themes)
        style = random.choice(self.styles)
        context = random.choice(self.contexts)
        return f"{theme} inside a {style} {self.product_name}, {context}"

# User input
product_name = input("Enter a product name for ad creation (e.g., 'PS5', 'AirPods', 'Kirkland Vodka'): ")

# Generate creative concept
prompt_generator = ProductPromptGenerator(product_name)
creative_prompt = prompt_generator.generate_prompt()

# Run tasks using Flow
concept_flow = Flow(llm=llm, max_loops=1, dashboard=False)
design_flow = Flow(llm=llm, max_loops=1, dashboard=False)
copywriting_flow = Flow(llm=llm, max_loops=1, dashboard=False)

# Execute tasks
concept_result = run_task("Generate a creative concept", product_name, concept_flow)
design_result = run_task("Suggest visual design ideas", product_name, design_flow)
copywriting_result = run_task("Create compelling ad copy for the product photo", product_name, copywriting_flow)

# Generate product image
image_paths = sd_api.run(creative_prompt)

# Output the results
print("Creative Concept:", concept_result)
print("Design Ideas:", design_result)
print("Ad Copy:", copywriting_result)
print("Image Path:", image_paths[0] if image_paths else "No image generated")
