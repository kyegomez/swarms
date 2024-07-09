import os
from swarms.models import OpenAIChat

# Load doten
openai = OpenAIChat(
    openai_api_key=os.getenv("OPENAI_API_KEY"), verbose=False
)

chat = openai("What are quantum fields?")
print(chat)
