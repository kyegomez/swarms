from swarms.utils.litellm_wrapper import LiteLLM

model = LiteLLM(
    model_name="groq/meta-llama/llama-4-scout-17b-16e-instruct",
    verbose=True,
)

print(model.run("What is your purpose in life?"))
