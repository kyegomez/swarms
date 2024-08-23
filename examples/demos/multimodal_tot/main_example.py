"""
Multi Modal tree of thoughts that leverages the GPT-4 language model and the
Stable Diffusion model to generate a multimodal output and evaluate the
output based a metric from 0.0 to 1.0 and then run a search algorithm using DFS and BFS and return the best output.


task: Generate an image of a swarm of bees -> Image generator -> GPT4V evaluates the img from 0.0 to 1.0 -> DFS/BFS -> return the best output


- GPT4Vision will evaluate the image from 0.0 to 1.0 based on how likely it accomplishes the task
- DFS/BFS will search for the best output based on the evaluation from GPT4Vision
- The output will be a multimodal output that is a combination of the image and the text
- The output will be evaluated by GPT4Vision
- The prompt to the image generator will be optimized from the output of GPT4Vision and the search

"""

import os

from dotenv import load_dotenv
from termcolor import colored

from swarms.models.gpt4_vision_api import GPT4VisionAPI
from swarms.models.stable_diffusion import StableDiffusion

# Load the environment variables
load_dotenv()

# Get the API key from the environment
api_key = os.environ.get("OPENAI_API_KEY")
stable_api_key = os.environ.get("STABLE_API_KEY")


# Initialize the language model
llm = GPT4VisionAPI(
    openai_api_key=api_key,
    max_tokens=500,
)

# IMG Generator
img_generator = StableDiffusion(api_key=stable_api_key)


# Initialize the language model
task = "Garden of Eden futuristic city graphic art"


def evaluate_img(llm, task: str, img: str):
    EVAL_IMG = f"""
    Evaluate the image: {img} on a scale from 0.0 to 1.0 based on how likely it accomplishes the task: {task}. Output nothing than the float representing the evaluated img.
    """
    out = llm.run(task=EVAL_IMG, img=img)
    out = float(out)
    return out


def enrichment_prompt(starting_prompt: str, evaluated_img: str):
    enrichment_task = (
        "Create a concise and effective image generation prompt"
        " within 400 characters or less, based on Stable Diffusion"
        " and Dalle best practices. Starting prompt:"
        f" \n\n'{starting_prompt}'\n\nImprove the prompt with any"
        " applicable details or keywords by considering the"
        " following aspects: \n1. Subject details (like actions,"
        " emotions, environment) \n2. Artistic style (such as"
        " surrealism, hyperrealism) \n3. Medium (digital painting,"
        " oil on canvas) \n4. Color themes and lighting (like warm"
        " colors, cinematic lighting) \n5. Composition and framing"
        " (close-up, wide-angle) \n6. Additional elements (like a"
        " specific type of background, weather conditions) \n7. Any"
        " other artistic or thematic details that can make the image"
        " more vivid and compelling. 8. Based on the evaluation of"
        " the first generated prompt used by the first prompt:"
        f" {evaluated_img} Enrich the prompt to generate a more"
        " compelling image. Output only a new prompt to create a"
        " better image"
    )
    return enrichment_task


# Main loop
max_iterations = 10  # Define the maximum number of iterations
best_score = 0
best_image = None

for _ in range(max_iterations):
    # Generate an image and get its path
    print(colored(f"Generating img for Task: {task}", "purple"))

    img_path = img_generator.run(
        task=task
    )  # This should return the file path of the generated image
    img_path = img_path[0]
    print(colored(f"Generated Image Path: {img_path}", "green"))

    # Evaluate the image by passing the file path
    score = evaluate_img(llm, task, img_path)
    print(
        colored(f"Evaluated Image Score: {score} for {img_path}", "cyan")
    )

    # Update the best score and image path if necessary
    if score > best_score:
        best_score = score
        best_image_path = img_path

    # Enrich the prompt based on the evaluation
    prompt = enrichment_prompt(task, score)
    print(colored(f"Enrichment Prompt: {prompt}", "yellow"))


# Output the best result
print("Best Image Path:", best_image_path)
print("Best Score:", best_score)
