<<<<<<< HEAD
from swarms import Agent, Fuyu

llm = Fuyu()

flow = Agent(max_loops="auto", llm=llm)
=======
from swarms import Flow, Fuyu

llm = Fuyu()

flow = Flow(max_loops="auto", llm=llm)
>>>>>>> 3d3dddaf0c7894ec2df14c51f7dd843c41c878c4

flow.run(
    task="Describe this image in a few sentences: ",
    img="https://unsplash.com/photos/0pIC5ByPpZY",
)
