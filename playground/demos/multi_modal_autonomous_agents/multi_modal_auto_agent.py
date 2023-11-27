<<<<<<< HEAD
from swarms.structs import Agent
=======
from swarms.structs import Flow
>>>>>>> 3d3dddaf0c7894ec2df14c51f7dd843c41c878c4
from swarms.models.gpt4_vision_api import GPT4VisionAPI


llm = GPT4VisionAPI()

task = "What is the color of the object?"
img = "images/swarms.jpeg"

## Initialize the workflow
<<<<<<< HEAD
flow = Agent(
=======
flow = Flow(
>>>>>>> 3d3dddaf0c7894ec2df14c51f7dd843c41c878c4
    llm=llm,
    max_loops="auto",
    dashboard=True,
)

flow.run(task=task, img=img)
