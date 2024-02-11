from swarms.structs import Agent
from swarms.models.gpt4_vision_api import GPT4VisionAPI


llm = GPT4VisionAPI()

task = "What is the color of the object?"
img = "images/swarms.jpeg"

## Initialize the workflow
agent = Agent(
    llm=llm,
    max_loops="auto",
    dashboard=True,
)

agent.run(task=task, img=img)
