from swarms import TogetherLLM

# Initialize the model with your parameters
model = TogetherLLM(
    model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
    max_tokens=1000,
)

# Run the model
model.run(
    "Generate a blog post about the best way to make money online."
)
