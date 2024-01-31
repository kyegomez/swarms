from swarms.models.openai_models import OpenAIChat

openai = OpenAIChat(openai_api_key="", verbose=False)

chat = openai("What are quantum fields?")
print(chat)
