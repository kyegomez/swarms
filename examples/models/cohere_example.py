from swarms.models import Cohere

cohere = Cohere(model="command-light", cohere_api_key="")

out = cohere("Hello, how are you?")
