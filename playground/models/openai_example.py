from swarms.models.openai_chat import OpenAIChat

model = OpenAIChat(openai_api_key="", openai_org_id="")

out = model("Hello, how are you?")

print(out)
