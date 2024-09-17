from swarm_models import vLLM

# Initialize vLLM with custom model and parameters
custom_vllm = vLLM(
    model_name="custom/model",
    tensor_parallel_size=8,
    trust_remote_code=True,
    revision="abc123",
    temperature=0.7,
    top_p=0.8,
)

# Generate text with custom configuration
generated_text = custom_vllm.run("Create a poem about nature.")
print(generated_text)
