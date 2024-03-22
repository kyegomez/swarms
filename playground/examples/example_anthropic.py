# Import necessary modules and classes
from swarms.models import Anthropic

# Initialize an instance of the Anthropic class
model = Anthropic(anthropic_api_key="")

# Using the run method
# completion_1 = model.run("What is the capital of France?")
# print(completion_1)

# Using the __call__ method
completion_2 = model(
    "How far is the moon from the earth?", stop=["miles", "km"]
)
print(completion_2)
