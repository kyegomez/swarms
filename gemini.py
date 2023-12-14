import os
from dotenv import load_dotenv
from swarms.models.gemini import Gemini

load_dotenv()

api_key = os.environ["GEMINI_API_KEY"]

# Initialize the model
model = Gemini(
    gemini_api_key="AIzaSyCUMFvG3TwwuZIniSpvDeP2AW5USyJUgJM"
)

# Establish the prompt and image
task = "What is your name"
img = "images/github-banner-swarms.png"

# Run the model
out = model.run("What is your name?", img=img)
print(out)
