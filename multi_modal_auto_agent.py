from swarms.structs import Flow
from swarms.models.gpt4_vision_api import GPT4VisionAPI


llm = GPT4VisionAPI()

task = "What is the color of the object?"
img = "images/swarms.jpeg"

## Initialize the workflow
flow = Flow(
    llm=llm,
    max_loops='auto',
    dashboard=True,
)

flow.run(task=task, img=img)
