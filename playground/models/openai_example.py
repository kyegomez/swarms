from swarms.models.openai_chat import OpenAIChat

model = OpenAIChat()

out = model("Hello, how are you?")

print(out)
