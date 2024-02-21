from swarms import Mistral

# Initialize the model
model = Mistral(
    model_name="miqudev/miqu-1-70b",
    max_length=500,
    use_flash_attention=True,
    load_in_4bit=True,
)

# Run the model
result = model.run("What is the meaning of life?")
