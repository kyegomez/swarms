# This might not work in the beginning but it's a starting point
<<<<<<< HEAD
from swarms.structs import Agent, GPT4V

llm = GPT4V()

flow = Agent(
=======
from swarms.structs import Flow, GPT4V

llm = GPT4V()

flow = Flow(
>>>>>>> 3d3dddaf0c7894ec2df14c51f7dd843c41c878c4
    max_loops="auto",
    llm=llm,
)

flow.run(
    task="Describe this image in a few sentences: ",
    img="https://unsplash.com/photos/0pIC5ByPpZY",
)
