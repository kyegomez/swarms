from swarms.utils.litellm_wrapper import LiteLLM

model = LiteLLM(model_name="gpt-4o-mini", verbose=True)

print(model.run("What is your purpose in life?"))
