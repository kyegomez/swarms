from swarms.models import Mixtral

# Initialize the Mixtral model with 4 bit and flash attention!
mixtral = Mixtral(load_in_4bit=True, use_flash_attention_2=True)

# Generate text for a simple task
generated_text = mixtral.run("Generate a creative story.")

# Print the generated text
print(generated_text)
