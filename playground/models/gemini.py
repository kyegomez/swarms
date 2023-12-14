from swarms.models.gemini import Gemini

# Initialize the model
model = Gemini(
    gemini_api_key="A",
)

# Establish the prompt and image
task = "What is your name"
img = "images/github-banner-swarms.png"

# Run the model
out = model.run("What is your name?", img)
print(out)
