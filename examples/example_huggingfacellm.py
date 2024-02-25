from swarms.models import HuggingfaceLLM

# Initialize with custom configuration
custom_config = {
    "quantize": True,
    "quantization_config": {"load_in_4bit": True},
    "verbose": True,
}
inference = HuggingfaceLLM(
    model_id="NousResearch/Nous-Hermes-2-Vision-Alpha", **custom_config
)

# Generate text based on a prompt
prompt_text = (
    "Create a list of known biggest risks of structural collapse with references"
)
generated_text = inference(prompt_text)
print(generated_text)
