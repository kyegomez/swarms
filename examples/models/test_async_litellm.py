from swarms.utils.litellm_wrapper import LiteLLM

llm = LiteLLM(
    model_name="gpt-4o-mini",
    temperature=0.5,
    max_tokens=1000,
    stream=True,
)

out = llm.run("What is the capital of France?")

print(out)
for chunk in out:
    out = chunk["choices"][0]["delta"]
    print(type(out))
    print(out)
