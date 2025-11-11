from swarms.utils.vllm_wrapper import VLLMWrapper

# Initialize the vLLM wrapper
vllm = VLLMWrapper(
    model_name="gpt-4o-mini",
    system_prompt="You are a helpful assistant.",
    temperature=0.7,
    max_tokens=4000
)

# Run inference
response = vllm.run("What is the capital of France?")
print(response)