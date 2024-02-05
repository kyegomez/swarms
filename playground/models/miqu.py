from swarms import Mistral


# Initialize the model
model = Mistral(
    model_name="mistralai/Mistral-7B-v0.1",
    max_length=500,
    use_flash_attention=True,
    load_in_4bit=True
)

# Run the model
result = model.run("What is the meaning of life?")