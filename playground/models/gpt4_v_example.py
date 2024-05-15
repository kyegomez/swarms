import os  # Import the os module for working with the operating system

from dotenv import (
    load_dotenv,  # Import the load_dotenv function from the dotenv module
)

from swarms import (
    GPT4VisionAPI,  # Import the GPT4VisionAPI class from the swarms module
)

# Load the environment variables
load_dotenv()

# Get the API key from the environment variables
api_key = os.getenv("OPENAI_API_KEY")

# Create an instance of the GPT4VisionAPI class with the API key and model name
gpt4vision = GPT4VisionAPI(
    openai_api_key=api_key,
    model_name="gpt-4o",
    max_tokens=1000,
    openai_proxy="https://api.openai.com/v1/chat/completions",
)

# Define the URL of the image to analyze
img = "ear.png"

# Define the task to perform on the image
task = "What is this image"

# Run the GPT4VisionAPI on the image with the specified task
answer = gpt4vision.run(task, img, return_json=True)

# Print the answer
print(answer)
