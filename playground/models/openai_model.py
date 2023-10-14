from swarms.models.openai_models import OpenAIChat

openai = OpenAIChat(openai_api_key="", verbose=False)

chat = openai("Are quantum fields everywhere?")
print(chat)
