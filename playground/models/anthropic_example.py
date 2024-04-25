from swarms.models import Anthropic

model = Anthropic(anthropic_api_key="")

task = "What is quantum field theory? What are 3 books on the field?"

print(model(task))
