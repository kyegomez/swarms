from swarms import Agent
from swarms.utils.image_file_b64 import get_image_data_uri

# Initialize agent
agent = Agent(
    model_name="gpt-4.1",
    max_loops=1,
    verbose=True,
)


# Example 2: Using a data URI (base64 with data URI prefix)
data_uri = get_image_data_uri("image.jpg")

task2 = "where does this image come from?"
result = agent.run(task=task2, img=data_uri)

print(result)
