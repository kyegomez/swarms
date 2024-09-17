import os

from dotenv import load_dotenv

from swarm_models.gemini import Gemini
from swarms.prompts.react import react_prompt

load_dotenv()

api_key = os.environ["GEMINI_API_KEY"]

# Establish the prompt and image
task = "What is your name"
img = "images/github-banner-swarms.png"

# Initialize the model
model = Gemini(
    gemini_api_key=api_key,
    model_name="gemini-pro",
    max_tokens=1000,
    system_prompt=react_prompt(task=task),
    temperature=0.5,
)


out = model.chat(
    "Create the code for a react component that displays a name",
    img=img,
)
print(out)
