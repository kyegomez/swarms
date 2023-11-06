from swarms.models.openai_models import OpenAIChat

openai = OpenAIChat(openai_api_key="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", verbose=False)

chat = openai("What are quantum fields?")
print(chat)
