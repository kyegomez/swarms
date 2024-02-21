# This might not work in the beginning but it's a starting point
from swarms.structs import GPT4V, Agent

llm = GPT4V()

agent = Agent(
    max_loops="auto",
    llm=llm,
)

agent.run(
    task="Describe this image in a few sentences: ",
    img="https://unsplash.com/photos/0pIC5ByPpZY",
)
