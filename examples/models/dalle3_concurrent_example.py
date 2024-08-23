"""

User task ->> GPT4 for prompt enrichment ->> Dalle3V for image generation
->> GPT4Vision for image captioning ->> Dalle3 better image

"""

import os

from swarms.models.dalle3 import Dalle3

api_key = os.environ["OPENAI_API_KEY"]

dalle3 = Dalle3(openai_api_key=api_key, n=1)

# task = "Swarm of robots working super industrial ambience concept art"

# image_url = dalle3(task)

tasks = ["A painting of a dog", "A painting of a cat"]
results = dalle3.process_batch_concurrently(tasks)

# print(results)
