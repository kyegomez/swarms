from swarms.models.anthropic import Anthropic


model = Anthropic(anthropic_api_key="")


task = "Say hello to"

print(model(task))
